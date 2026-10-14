"""Launch the installed profile with resource and provenance checks."""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import psutil

from hardware import resource_info, adapt
from runtime import (
    check_build,
    check_config,
    engine_environment,
    read_json,
    profile_lock,
    require_free_port,
    stop_owned,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the launcher without loading the model",
    )
    options = parser.parse_args()
    path = options.profile.resolve()
    if not path.is_file():
        raise RuntimeError(
            "Installed profile is missing. Run INSTALL.cmd with the current model folder."
        )
    with profile_lock(path.parent):
        return run_profile(path, options.check)


def run_profile(path, check=False):
    cfg = read_json(path)
    require_free_port(cfg["host"], cfg["port"])
    hardware = resource_info(cfg.get("_recipe_gpu_index", 0))
    build = check_build(hardware)
    check_config(cfg)
    cfg, values, selection = adapt(
        cfg,
        read_json(path.with_suffix(".env.json")),
        hardware,
        cfg.get("_recipe_workers"),
        cfg.get("_recipe_tuning"),
    )
    env = engine_environment(values, build)
    if check:
        print("PASS: model, pack, source, build, port and resource checks.", flush=True)
        return 0
    launch_config = path.with_name(path.stem + ".launch.json")
    write_json(launch_config, cfg)
    print("Resource settings:", json.dumps(selection), flush=True)
    auth = (
        "key file: " + str(path.parent / "api-key.txt")
        if cfg.get("api_key")
        else "no API key required"
    )
    print(f"API: http://127.0.0.1:{cfg['port']}/v1 ; {auth}", flush=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "serve.server",
            "--engine",
            "strata",
            "--config",
            str(launch_config),
            "--host",
            cfg["host"],
            "--port",
            str(cfg["port"]),
        ],
        cwd=cfg["cwd"],
        env=env,
    )
    seen = {}
    try:
        seen[proc.pid] = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        pass
    try:
        while proc.poll() is None:
            try:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    seen[child.pid] = child.create_time()
            except psutil.Error:
                pass
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={hardware['gpu']['uuid']}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                used = int(output.splitlines()[0])
            except (
                ValueError,
                IndexError,
                OSError,
                subprocess.SubprocessError,
            ) as error:
                raise RuntimeError(
                    "GPU telemetry unavailable; stopped the owned model."
                ) from error
            if used > selection["gpu_limit_mib"]:
                raise RuntimeError(
                    f"Total GPU usage {used} MiB exceeded the memory guard. Close GPU-heavy "
                    "applications or increase --vram-reserve-mib before restarting."
                )
            if psutil.virtual_memory().available / 1048576 < selection["ram_floor_mib"]:
                raise RuntimeError(
                    "Available RAM fell below 512 MiB; stopped the owned model. Close "
                    "memory-heavy applications or reduce --ram-experts-gib."
                )
            time.sleep(0.5)
    finally:
        stop_owned(proc, seen)
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print("Launch stopped:", error, file=sys.stderr)
        sys.exit(1)
