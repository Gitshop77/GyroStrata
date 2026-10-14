"""Measure installed production binaries serially; preserve launchers/configs/weights."""

import marshal
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import psutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hardware import resource_info, adapt
from runtime import (
    check_build,
    check_config,
    profile_lock,
    stop_owned as stop_process_tree,
)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
WARM_8K = True
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--out", default="local-results")
ap.add_argument("--models", nargs="+", choices=["M"], default=["M"])
options = ap.parse_args()
MODEL_DIR = pathlib.Path(options.model_dir).resolve()
OUT = pathlib.Path(options.out).resolve()

OUT.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / ".venv/Scripts/python.exe"
PORT = 8081
TARGETS = [8192, 32768, 65536, 131072, 196608, 261000]

MANIFEST = []
for name in options.models:
    cp = MODEL_DIR / ".gyro-recipe" / f"{name}.json"
    c = json.loads(cp.read_text(encoding="utf-8-sig"))
    MANIFEST.append(
        dict(
            model=c["model_name"],
            config=str(cp),
            launcher=str(MODEL_DIR / f"Start-Gyro-{name}-256K.cmd"),
            engine_sha256=hashlib.sha256(
                pathlib.Path(c["exe"]).read_bytes()
            ).hexdigest(),
            context=262144,
            vision=False,
        )
    )


def write(path, value):
    if path.name.endswith(".results.json"):
        value = dict(value)
        value.pop("source_paths", None)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def launcher_environment(path):
    name = "M"
    env = os.environ.copy()
    for k, v in json.loads(
        (MODEL_DIR / ".gyro-recipe" / f"{name}.env.json").read_text(
            encoding="utf-8-sig"
        )
    ).items():
        if v:
            env[k] = v
        else:
            env.pop(k, None)
    env["CUDA_PATH"] = json.loads(
        (ROOT / "build-info.json").read_text(encoding="utf-8-sig")
    )["cuda_root"]
    if env.get("CUDA_PATH"):
        cuda = pathlib.Path(env["CUDA_PATH"])
        env["PATH"] = (
            str(cuda / "bin/x64")
            + os.pathsep
            + str(cuda / "bin")
            + os.pathsep
            + env.get("PATH", "")
        )
    return env


def stop_owned(proc, seen):
    if proc is not None:
        stop_process_tree(proc, seen)


def run_model(item):
    model = item["model"]
    saved = OUT / (model + ".results.json")
    original = pathlib.Path(item["config"])
    cfg = json.loads(original.read_text(encoding="utf-8-sig"))
    hw = resource_info(cfg.get("_recipe_gpu_index", 0))
    check_build(hw)
    check_config(cfg)
    name = "M"
    cfg, selected_env, resource_selection = adapt(
        cfg,
        json.loads(
            (MODEL_DIR / ".gyro-recipe" / f"{name}.env.json").read_text(
                encoding="utf-8-sig"
            )
        ),
        hw,
        cfg.get("_recipe_workers"),
        cfg.get("_recipe_tuning"),
    )
    limit = resource_selection["gpu_limit_mib"]
    profile_index = cfg["args"].index("--expert-profile") + 1
    profile_original = pathlib.Path(cfg["args"][profile_index])
    identities = {
        str(p): sha(p)
        for p in [
            original,
            pathlib.Path(item["launcher"]),
            profile_original,
            pathlib.Path(cfg["exe"]),
        ]
    }
    require(
        identities[cfg["exe"]] == item["engine_sha256"],
        "Installed binary differs from final manifest",
    )
    environment = launcher_environment(item["launcher"])
    for key, value in selected_env.items():
        if value:
            environment[key] = value
        else:
            environment.pop(key, None)
    evidence = dict(
        version=2,
        files=identities,
        targets=TARGETS,
        warmup=WARM_8K,
        protocol=hashlib.sha256(marshal.dumps(run_model.__code__)).hexdigest(),
        resource_selection=resource_selection,
        selected_args=cfg["args"],
        environment={
            k: v
            for k, v in environment.items()
            if k.startswith(("POLY_", "STRATA_", "OMP_", "MKL_", "KMP_"))
        },
    )
    fingerprint = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode()
    ).hexdigest()
    if saved.exists():
        prior = json.loads(saved.read_text(encoding="utf-8-sig"))
        if prior.get("status") == "passed":
            if prior.get("evidence_fingerprint") != fingerprint:
                raise RuntimeError(
                    "Saved benchmark provenance differs or is missing; select a fresh output directory. Historical evidence was preserved."
                )
            print("SKIP VERIFIED COMPLETED", model, flush=True)
            return prior

    profile_copy = OUT / (model + profile_original.suffix)
    shutil.copy2(profile_original, profile_copy)
    cfg["args"][profile_index] = str(profile_copy)
    for name in list(cfg):
        if name.startswith("expert_profile_save"):
            cfg.pop(name)
    key = secrets.token_urlsafe(32)
    cfg.update(
        host="127.0.0.1",
        port=PORT,
        log=str(OUT / (model + ".engine.log")),
        api_key=key,
        open_browser=False,
        api_monitor=False,
    )
    config_path = OUT / (model + ".config.json")
    write(config_path, cfg)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    url = f"http://127.0.0.1:{PORT}"
    env = environment.copy()
    env["PYTHONUNBUFFERED"] = "1"
    spec = importlib.util.spec_from_file_location(
        "sweep_tokenizer", pathlib.Path(cfg["cwd"]) / "tools/strata_tokenizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = (
        module.Tokenizer.from_dir(pathlib.Path(cfg["tokenizer"]))
        if hasattr(module.Tokenizer, "from_dir")
        else module.Tokenizer.from_gguf(cfg["args"][cfg["args"].index("--native") + 1])
    )
    corpus = "\n".join(
        f"def operation_{i}(value): return (value * {i + 3} + {i + 17}) % 1000003"
        for i in range(12000)
    )
    corpus_ids = tokenizer.encode(corpus)
    require(len(corpus_ids) > max(TARGETS), "Benchmark corpus is too short")
    nonce = secrets.token_hex(6)
    result = dict(
        model=model,
        context_allocation=262144,
        vision_enabled=item["vision"],
        engine_sha256=item["engine_sha256"],
        evidence_fingerprint=fingerprint,
        sampling=cfg["sampling"],
        resource_selection=resource_selection,
        initial_available_ram_mib=round(hw["available_ram_gib"] * 1024, 1),
        initial_gpu_used_mib=hw["gpu"]["used_mib"],
        decode_tokens=256,
        cached_repetitions=3,
        stages=[],
        status="starting",
        source_paths=dict(config=str(original), launcher=item["launcher"]),
        note="Text workloads; thinking disabled only per benchmark request; production default unchanged.",
    )
    proc = None
    seen = {}
    stopped = threading.Event()
    state = dict(
        peak_vram_mib=0,
        min_available_ram_mib=999999,
        peak_temperature_c=0,
        over_limit=False,
        samples=0,
        telemetry_error=None,
    )
    stage_memory = dict(
        peak_vram_mib=0, min_available_ram_mib=999999, peak_temperature_c=0, samples=0
    )
    lock = threading.Lock()

    def monitor():
        while not stopped.is_set():
            try:
                values = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id=" + env.get("CUDA_VISIBLE_DEVICES", "0"),
                        "--query-gpu=memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=8,
                )
                used, temp = [int(v.strip()) for v in values.splitlines()[0].split(",")]
                available = psutil.virtual_memory().available / 1048576
                with lock:
                    for record in (state, stage_memory):
                        record["peak_vram_mib"] = max(record["peak_vram_mib"], used)
                        record["peak_temperature_c"] = max(
                            record["peak_temperature_c"], temp
                        )
                        record["min_available_ram_mib"] = min(
                            record["min_available_ram_mib"], available
                        )
                    state["samples"] += 1
                    stage_memory["samples"] += 1
                    if used > limit:
                        state["over_limit"] = True
                if proc is not None and proc.poll() is None:
                    try:
                        parent = psutil.Process(proc.pid)
                        seen[parent.pid] = parent.create_time()
                        for child in parent.children(recursive=True):
                            seen[child.pid] = child.create_time()
                    except psutil.Error:
                        pass
                if used > limit:
                    stop_owned(proc, seen)
                    break
            except (
                OSError,
                ValueError,
                IndexError,
                subprocess.SubprocessError,
                psutil.Error,
            ) as error:
                with lock:
                    state["telemetry_error"] = f"{type(error).__name__}: {error}"
                stop_owned(proc, seen)
                break
            stopped.wait(0.5)

    def http(route):
        with urllib.request.urlopen(
            urllib.request.Request(url + route, headers=headers), timeout=3
        ) as response:
            return json.load(response)

    def prompt(target):
        prefix = f"Sweep document {nonce}-{target}. Secret marker: ORBIT-7319.\n"
        suffix = "\nFirst output the secret marker from the beginning. Then continue this module with at least twenty more Python functions in the same style. Start immediately; no explanation."
        limit = target - len(tokenizer.encode(prefix + suffix)) - 32
        text = prefix + tokenizer.decode(corpus_ids[:limit]) + suffix
        # Decoding/re-encoding can change token boundaries; correct the document length.
        for _ in range(4):
            difference = (target - 32) - len(tokenizer.encode(text))
            if abs(difference) <= 2:
                break
            limit += difference
            text = prefix + tokenizer.decode(corpus_ids[:limit]) + suffix
        return text

    def chat(text, validate_output=True, **override):
        body = dict(
            model=model,
            messages=[dict(role="user", content=text)],
            max_tokens=256,
            seed=42,
            stream=True,
            stream_options=dict(include_usage=True),
            chat_template_kwargs=dict(
                enable_thinking=False, preserve_thinking=cfg["preserve_thinking"]
            ),
        )
        body.update({k: v for k, v in cfg["sampling"].items() if k != "penalty_last_n"})
        body.update(override)
        started = time.perf_counter()
        first = None
        content, reasoning, timings, usage = "", "", {}, {}
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                if raw.strip() == b"data: [DONE]":
                    break
                event = json.loads(raw[6:])
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                timings = event.get("timings") or timings
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text_part = delta.get("content") or ""
                    think_part = delta.get("reasoning_content") or ""
                    if (text_part or think_part) and first is None:
                        first = time.perf_counter() - started
                    content += text_part
                    reasoning += think_part
        answer = dict(
            ttft_s=first,
            wall_s=time.perf_counter() - started,
            content=content,
            reasoning=reasoning,
            timings=timings,
            usage=usage,
        )
        engine_text = (OUT / (model + ".engine.log")).read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(r"non-finite logits|DONE nonfinite", engine_text):
            raise RuntimeError(
                "Non-finite engine result detected, including a recovered/retried request."
            )
        if state["telemetry_error"]:
            raise RuntimeError(
                "GPU/RAM telemetry failed; benchmark stopped: "
                + state["telemetry_error"]
            )
        if state["samples"] < 1:
            raise RuntimeError(
                "No successful GPU/RAM telemetry samples; benchmark is invalid."
            )
        require(
            not validate_output or "ORBIT-7319" in content,
            "Context marker retrieval failed",
        )
        require(
            not validate_output or timings.get("predicted_n", 0) >= 240,
            "Output ended too early for decode measurement",
        )
        require(not state["over_limit"], "GPU/RAM limit exceeded")
        return answer

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", PORT))
        log = (OUT / (model + ".server.log")).open("w", encoding="utf-8")
        started = time.perf_counter()
        proc = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "serve.server",
                "--engine",
                "strata",
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            cwd=cfg["cwd"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        thread.start()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if state["telemetry_error"]:
                raise RuntimeError(
                    "GPU/RAM telemetry failed; benchmark stopped: "
                    + state["telemetry_error"]
                )
            if proc.poll() is not None:
                raise RuntimeError("Server exited during startup; inspect saved log")
            try:
                health = http("/health")
                if health.get("loaded"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Startup timeout")
        require(
            health["model"] == model and health["max_context"] == 262144,
            "Server model/context differs from the profile",
        )
        require(
            bool(health.get("images")) == item["vision"],
            "Server vision configuration differs",
        )
        result["startup_s"] = time.perf_counter() - started
        result["status"] = "running"
        print("READY", model, round(result["startup_s"], 2), flush=True)
        write(saved, result)
        if WARM_8K:
            result["warmup"] = chat(prompt(4096), temperature=0)
            write(saved, result)
            print("WARMUP DONE", model, flush=True)
        for target in TARGETS:
            with lock:
                stage_memory.update(
                    peak_vram_mib=0,
                    min_available_ram_mib=999999,
                    peak_temperature_c=0,
                    samples=0,
                )
            text = prompt(target)
            stage = dict(target_tokens=target, cold=None, cached=[], status="running")
            result["stages"].append(stage)
            print("START", model, target, flush=True)
            write(saved, result)
            cold = chat(text)
            require(
                cold["timings"].get("cache_n", 0) < 256,
                "Cold test reused a significant prompt prefix",
            )
            count = cold["usage"]["prompt_tokens"]
            require(
                abs(count - target) <= 256,
                f"Unexpected prompt length: {count} vs {target}",
            )
            stage["prompt_tokens"] = count
            stage["cold"] = cold
            print(
                "COLD",
                model,
                target,
                "actual",
                count,
                "pp",
                cold["timings"]["prompt_per_second"],
                "decode",
                cold["timings"]["predicted_per_second"],
                flush=True,
            )
            write(saved, result)
            for repeat in range(3):
                cached = chat(text)
                require(
                    cached["timings"].get("cache_n", 0) >= count - 32,
                    "Cached test lost the prompt prefix",
                )
                stage["cached"].append(cached)
                print(
                    "CACHED",
                    model,
                    target,
                    repeat,
                    cached["timings"]["predicted_per_second"],
                    flush=True,
                )
                write(saved, result)
            speeds = [
                answer["timings"]["predicted_per_second"] for answer in stage["cached"]
            ]
            times = [answer["timings"]["predicted_ms"] for answer in stage["cached"]]
            tokens = [answer["timings"]["predicted_n"] for answer in stage["cached"]]
            stage["summary"] = dict(
                prefill_tps=cold["timings"]["prompt_per_second"],
                output_tps=sum(
                    a["timings"]["predicted_n"] for a in [cold, *stage["cached"]]
                )
                / (
                    sum(a["timings"]["predicted_ms"] for a in [cold, *stage["cached"]])
                    / 1000
                ),
                first_reply_output_tps=cold["timings"]["predicted_per_second"],
                cached_decode_tps=sum(tokens) / (sum(times) / 1000),
                cached_decode_min=min(speeds),
                cached_decode_max=max(speeds),
                cold_ttft_s=cold["ttft_s"],
                cached_ttft_s=sum(x["ttft_s"] for x in stage["cached"])
                / len(stage["cached"]),
            )
            with lock:
                stage["memory"] = stage_memory.copy()
            if stage["memory"]["samples"] < 1:
                raise RuntimeError(
                    "No successful GPU/RAM telemetry samples during this context stage."
                )
            stage["status"] = "passed"
            write(saved, result)
            print("STAGE DONE", model, target, json.dumps(stage["summary"]), flush=True)
        short = chat(
            "What is 17 + 25? Reply with only the integer.",
            validate_output=False,
            max_tokens=32,
            temperature=0,
        )
        result["post_context"] = dict(short=short)
        write(saved, result)
        require("42" in short["content"], "Post-context arithmetic failed")
        returned = chat(text)
        result["post_context"]["returned"] = returned
        write(saved, result)
        require(
            returned["timings"].get("cache_n", 0) >= count - 32,
            "Conversation return lost the saved full-context prefix",
        )
        engine_text = (OUT / (model + ".engine.log")).read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(r"non-finite logits|DONE nonfinite", engine_text):
            raise RuntimeError(
                "Non-finite engine result detected, including a recovered/retried request; not a clean stability pass."
            )
        result["nonfinite_log_check"] = "passed"
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error))
        print("FAILED", model, str(error), flush=True)
    finally:
        stop_owned(proc, seen)
        stopped.set()
        if thread.is_alive():
            thread.join(10)
        if "log" in locals():
            log.close()
        result["memory"] = state
        result["production_files_unchanged"] = all(
            sha(p) == digest for p, digest in identities.items()
        )
        if not result["production_files_unchanged"]:
            result.update(
                status="failed", error="A production file changed during the test"
            )
        cfg.pop("api_key", None)
        write(config_path, cfg)
        write(saved, result)
        print("MODEL DONE", model, result["status"], json.dumps(state), flush=True)
    return result


def report(results):
    def public(value):
        if isinstance(value, dict):
            return {
                k: public(v)
                for k, v in value.items()
                if k not in ("source_paths", "api_key", "files", "selected_args")
            }
        if isinstance(value, list):
            return [public(v) for v in value]
        return value

    write(
        OUT / "summary.json",
        dict(
            models=public(results),
            targets=TARGETS,
            status="passed"
            if len(results) == len(MANIFEST)
            and all(r["status"] == "passed" for r in results)
            else "incomplete",
        ),
    )


if __name__ == "__main__":
    results = []
    for item in MANIFEST:
        with profile_lock(pathlib.Path(item["config"]).parent):
            results.append(run_model(item))
        report(results)
        if results[-1]["status"] != "passed":
            raise SystemExit(1)
    print("ALL SELECTED GYRO MODELS PASSED", flush=True)
