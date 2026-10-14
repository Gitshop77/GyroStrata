"""Pinned Windows source builder and local profile installer. Never downloads weights."""

import argparse, datetime, hashlib, json, os, pathlib, re, secrets, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8-sig"))


def run(*args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def configure_api_key(runtime, no_api_key):
    keypath = runtime / "api-key.txt"
    if no_api_key:
        keypath.unlink(missing_ok=True)
        for path in runtime.rglob("*"):
            # Auth cleanup must never edit tokenizer vocabulary or pack metadata.
            if not path.is_file() or not re.fullmatch(
                r"M(?:\.launch|\.config)?\.json(?:\.previous|\.tmp)?", path.name
            ):
                continue
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(config, dict) and "api_key" in config:
                from runtime import write_json

                config.pop("api_key")
                write_json(path, config)
        return ""
    if not keypath.exists():
        keypath.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    key = keypath.read_text(encoding="utf-8").strip()
    if len(key) < 24:
        raise RuntimeError(
            "Existing API key is too short; preserve it and supply a key of at least 24 characters."
        )
    return key


def configure_install_api_key(runtime, api_key=False):
    return configure_api_key(runtime, not api_key)


def sources(args):
    if sha(HERE / "patches/strata.patch") != PINS["strata"]["patch_sha256"]:
        raise RuntimeError("Patch hash mismatch")
    for name in ["strata", "llama"]:
        pin = PINS[name]
        dst = HERE / "sources" / name
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / ".git").exists():
            if any(dst.iterdir()):
                raise RuntimeError(
                    "Source directory is not an owned Git checkout: " + str(dst)
                )
            run("git", "init", dst)
        run("git", "-C", dst, "config", "core.longpaths", "true")
        run("git", "-C", dst, "config", "core.autocrlf", "false")
        head = subprocess.run(
            ["git", "-C", str(dst), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.returncode:
            # A failed download after git init is safe to retry; no reset or restore.
            if subprocess.check_output(
                ["git", "-C", str(dst), "status", "--porcelain"], text=True
            ).strip():
                raise RuntimeError(
                    "Interrupted source checkout contains user files; preserve it and use a fresh engine folder."
                )
            run(
                "git",
                "-C",
                dst,
                "fetch",
                "--depth",
                "1",
                getattr(args, name + "_source") or pin["url"],
                pin["revision"],
            )
            run("git", "-C", dst, "checkout", "--detach", "FETCH_HEAD")
        elif head.stdout.strip() != pin["revision"]:
            raise RuntimeError(
                "Existing source revision differs; use a fresh engine folder instead of overwriting it."
            )
        if name == "strata":
            check = subprocess.run(
                [
                    "git",
                    "-C",
                    str(dst),
                    "apply",
                    "--check",
                    str(HERE / "patches/strata.patch"),
                ],
                capture_output=True,
            )
            if check.returncode == 0:
                run(
                    "git",
                    "-C",
                    dst,
                    "apply",
                    "--whitespace=nowarn",
                    HERE / "patches/strata.patch",
                )
        changed = subprocess.check_output(
            ["git", "-C", str(dst), "diff", "HEAD", "--name-only"], text=True
        ).splitlines()
        untracked = subprocess.check_output(
            ["git", "-C", str(dst), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
        allowed = set(pin.get("patched_file_sha256", {}))
        if (set(changed) | set(untracked)) - allowed:
            raise RuntimeError(
                "Source contains unrelated modifications; preserve them and use a fresh engine folder."
            )
    for relative, expected in PINS["strata"]["patched_file_sha256"].items():
        path = HERE / "sources/strata" / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            != expected
        ):
            raise RuntimeError(
                "Patched source differs: " + relative + "; local files were preserved."
            )
    print(
        "Pinned sources ready and patch hashes verified. Weights were not downloaded."
    )


def validate_pack(pack, gguf):
    """Check quant offsets and Hadamard metadata against the exact target header."""
    for relative in [
        "dense.bin",
        "index.txt",
        "native_experts.txt",
        "tokenizer/vocab.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/token_type.json",
        "tokenizer/tokenizer.json",
    ]:
        if not (pack / relative).is_file():
            raise RuntimeError("Incomplete pack: " + str(pack / relative))
    sys.path.insert(0, str(HERE / "sources/strata/tools"))
    import iq_pack as iq

    model = iq.Model(gguf)
    got = iq.expert_layout(model, gguf)
    if isinstance(got, str):
        raise RuntimeError(got)
    layout, text, _, offset = got
    t0 = layout[0][5]
    had = iq.hadamard_spec(
        iq.G.GGUFFile(gguf).metadata, layout, int(t0[0].shape[0]), int(t0[2].shape[0])
    )
    if had is not None:
        if not had.startswith("# strata hadamard"):
            raise RuntimeError(had)
        text = iq.native_experts_v5(text)
        if (
            not (pack / "hadamard.txt").is_file()
            or (pack / "hadamard.txt").read_text(encoding="utf-8") != had
        ):
            raise RuntimeError("Pack Hadamard metadata differs from target GGUF.")
    if (pack / "native_experts.txt").read_text(encoding="utf-8") != text:
        raise RuntimeError("Pack native quant layout differs from target GGUF.")
    import strata_tokenizer

    tokenizer = strata_tokenizer.Tokenizer.from_gguf(gguf)
    token_path = pack / "tokenizer"
    if (
        json.loads((token_path / "vocab.json").read_text(encoding="utf-8"))
        != tokenizer.ids
    ):
        raise RuntimeError(
            "Pack tokenizer vocabulary differs from the model. Preserve the pack and reinstall into a fresh profile directory."
        )
    if (token_path / "chat_template.jinja").read_bytes().decode(
        "utf-8"
    ) != iq.G.GGUFFile(gguf).metadata.get("tokenizer.chat_template"):
        raise RuntimeError("Pack chat template differs from the model.")
    if (
        json.loads((token_path / "token_type.json").read_text(encoding="utf-8"))
        != tokenizer.token_types
    ):
        raise RuntimeError("Pack special-token types differ from the model.")
    merges = "\n".join(
        "%s %s" % pair
        for pair, _ in sorted(tokenizer.ranks.items(), key=lambda item: item[1])
    )
    if (token_path / "merges.txt").read_text(encoding="utf-8") != merges:
        raise RuntimeError("Pack tokenizer merges differ from the model.")
    details = json.loads((token_path / "tokenizer.json").read_text(encoding="utf-8"))
    if (
        details.get("vocab_size") != len(tokenizer.tokens)
        or details.get("n_merges") != len(tokenizer.ranks)
        or details.get("special_ids") != tokenizer.special_ids
        or details.get("pre_pattern") != strata_tokenizer.QWEN35_PATTERN
        or details.get("add_bos_token") is not False
    ):
        raise RuntimeError("Pack tokenizer settings differ from the model.")
    if (pack / "experts.bin").exists():
        sidecar = pack / "experts.bin.src.json"
        if (
            not sidecar.is_file()
            or json.loads(sidecar.read_text(encoding="utf-8-sig"))
            != iq.experts_source(model, text, offset)
            or (pack / "experts.bin").stat().st_size != offset
        ):
            raise RuntimeError(
                "External expert pack has no matching target provenance."
            )


def install(args):
    from runtime import profile_lock, require_free_port

    require_free_port("0.0.0.0", 8080)
    profiledir = pathlib.Path(args.profile_dir or args.model_dir).resolve()
    with profile_lock(profiledir / ".gyro-recipe"):
        return install_locked(args)


def launcher_text(python, entry, name="M"):
    # Static batch paths must escape percent signs; delayed expansion corrupts !.
    quote = lambda path: '"' + str(path).replace("%", "%%") + '"'
    return (
        "@echo off\nchcp 65001 >nul\nsetlocal DisableDelayedExpansion\n"
        + quote(python)
        + " "
        + quote(entry)
        + ' "%~dp0.gyro-recipe\\'
        + name
        + '.json" %*\nset "GYRO_EXIT=%errorlevel%"\n'
        + 'if not "%GYRO_EXIT%"=="0" if not defined GYRO_NO_PAUSE pause\nexit /b %GYRO_EXIT%\n'
    )


def install_locked(args):
    from runtime import write_json

    modeldir = pathlib.Path(args.model_dir).resolve()
    profiledir = (
        pathlib.Path(args.profile_dir).resolve() if args.profile_dir else modeldir
    )
    runtime = profiledir / ".gyro-recipe"
    py = HERE / ".venv/Scripts/python.exe"
    engine = HERE / "build-engine/strata.exe"
    if not py.is_file() or not engine.is_file():
        raise RuntimeError("Run build.cmd first")
    from hardware import resource_info, adapt
    from runtime import check_build

    hardware = resource_info(args.gpu)
    check_build(hardware)
    tuning = {
        key: getattr(args, key)
        for key in [
            "pcie_fraction",
            "pool_tasks",
            "ram_experts_gib",
            "vram_reserve_mib",
        ]
        if getattr(args, key, None) is not None
    }
    # Validate all selected inputs before changing an installed launcher.
    for name in args.models:
        template = json.loads(
            (HERE / "profiles" / f"{name}.json").read_text(encoding="utf-8-sig")
        )
        adapt(template, {}, hardware, args.workers, tuning)
        filenames = [template["_weight_file"]]
        for filename in filenames:
            weight = next(
                x for x in PINS["models"]["files"] if x["rfilename"] == filename
            )
            path = modeldir / filename
            if not path.is_file() or path.stat().st_size != weight["size"]:
                raise RuntimeError("Missing file or wrong size: " + filename)
            if not args.skip_weight_hash and sha(path) != weight["lfs"]["sha256"]:
                raise RuntimeError("Wrong SHA256: " + filename)
    runtime.mkdir(parents=True, exist_ok=True)
    backup = (
        runtime
        / "rollback"
        / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    for name in args.models:
        for path in [
            runtime / f"{name}.json",
            runtime / f"{name}.env.json",
            runtime / f"{name}.hardware.json",
            profiledir / f"Start-Gyro-{name}-256K.cmd",
        ]:
            if path.is_file():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup / path.name)
    for name in args.models:
        template = json.loads(
            (HERE / "profiles" / f"{name}.json").read_text(encoding="utf-8-sig")
        )
        filename = template.pop("_weight_file")
        gguf = modeldir / filename
        weight = next(x for x in PINS["models"]["files"] if x["rfilename"] == filename)
        supplied = getattr(args, "pack_" + name.lower())
        pack = (
            pathlib.Path(supplied).resolve()
            if supplied
            else runtime / f"pack-{name}-interleaved-v1"
        )
        if not supplied:
            marker = pack / "recipe-weight-sha.txt"
            if not marker.exists():
                pending = pack / "recipe-packing.json"
                if pack.exists() and (
                    not pending.is_file()
                    or json.loads(pending.read_text(encoding="utf-8-sig")).get(
                        "weight_sha256"
                    )
                    != weight["lfs"]["sha256"]
                ):
                    raise RuntimeError(
                        "Existing pack has no matching installation provenance; use --pack-m for an independently trusted pack, or a fresh profile directory."
                    )
                sys.path.insert(0, str(HERE / "sources/strata/tools"))
                import iq_pack as iq

                geometry = iq.expert_layout(iq.Model(gguf), gguf)
                if isinstance(geometry, str):
                    raise RuntimeError(geometry)
                needed = geometry[3] + 3 * 2**30
                if shutil.disk_usage(runtime).free < needed:
                    raise RuntimeError(
                        "Insufficient SSD space for interleaved experts and dense pack; need approximately "
                        + str(round(needed / 2**30, 1))
                        + " GiB free. Existing installation was preserved."
                    )
                pack.mkdir(parents=True, exist_ok=True)
                pending.write_text(
                    json.dumps(dict(weight_sha256=weight["lfs"]["sha256"]))
                )
                env = os.environ.copy()
                env["STRATA_GGUF_PY"] = str(HERE / "sources/llama/gguf-py")
                run(
                    py,
                    HERE / "sources/strata/tools/iq_pack.py",
                    "--gguf",
                    gguf,
                    "--out",
                    pack,
                    "--compat-bf16",
                    "--experts-bin",
                    env=env,
                )
                validate_pack(pack, gguf)
                marker.write_text(weight["lfs"]["sha256"])
                pending.unlink()
            if (
                marker.read_text(encoding="utf-8-sig").strip()
                != weight["lfs"]["sha256"]
            ):
                raise RuntimeError("Pack provenance differs from target weights.")
            if not (pack / "experts.bin").is_file():
                raise RuntimeError(
                    "Interleaved expert pack is incomplete; preserve it and rerun packing before launch."
                )
        validate_pack(pack, gguf)
        cuda = pathlib.Path(
            json.loads((HERE / "build-info.json").read_text(encoding="utf-8-sig"))[
                "cuda_root"
            ]
        )
        parity_env = os.environ.copy()
        parity_env["PATH"] = (
            str(cuda / "bin/x64")
            + os.pathsep
            + str(cuda / "bin")
            + os.pathsep
            + parity_env.get("PATH", "")
        )
        parity_env["CUDA_VISIBLE_DEVICES"] = hardware["gpu"]["uuid"]
        run(HERE / "build-engine/native_expert_parity.exe", gguf, env=parity_env)
        ranking = runtime / f"expert-{name}.bin"
        if sha(HERE / "data" / ranking.name) != PINS["expert_profiles"][name]["sha256"]:
            raise RuntimeError(
                "Expert seed hash mismatch. Extract a fresh release; installed authentication was left untouched."
            )
        shutil.copy2(HERE / "data" / ranking.name, ranking)
        key = configure_install_api_key(runtime, args.api_key)
        replacements = {
            "@SOURCE@": str(HERE / "sources/strata"),
            "@ENGINE@": str(engine),
            "@PACK@": str(pack),
            "@WEIGHT@": str(gguf),
            "@PROFILE@": str(ranking),
            "@LOG@": str(runtime / f"{name}.engine.log"),
        }

        def resolve(value):
            if isinstance(value, str):
                for before, after in replacements.items():
                    value = value.replace(before, after)
            elif isinstance(value, list):
                value = [resolve(x) for x in value]
            elif isinstance(value, dict):
                value = {k: resolve(v) for k, v in value.items()}
            return value

        cfg = resolve(template)
        if key:
            cfg["api_key"] = key
        else:
            cfg.pop("api_key", None)
        environment = json.loads(
            (HERE / "profiles" / f"{name}.env.json").read_text(encoding="utf-8-sig")
        )
        cfg, environment, selection = adapt(
            cfg, environment, hardware, args.workers, tuning
        )
        cfg["_recipe_gpu_index"] = args.gpu
        cfg["_recipe_workers"] = args.workers
        cfg["_recipe_tuning"] = tuning
        cfg["_recipe_weight_mtime_ns"] = gguf.stat().st_mtime_ns
        cfg["_recipe_pack_files"] = {
            str(path.relative_to(pack)).replace("\\", "/"): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in pack.rglob("*")
            if path.is_file() and path.name not in ("recipe-packing.json",)
        }
        output = runtime / f"{name}.json"
        if output.exists():
            shutil.copy2(output, output.with_suffix(".json.previous"))
        write_json(output, cfg)
        envfile = runtime / f"{name}.env.json"
        write_json(envfile, environment)
        write_json(runtime / f"{name}.hardware.json", selection)
        cmd = profiledir / f"Start-Gyro-{name}-256K.cmd"
        if cmd.exists():
            shutil.copy2(cmd, cmd.with_suffix(".cmd.previous"))
        cmd.write_text(launcher_text(py, HERE / "launch.py", name), encoding="utf-8")
        print(
            "Installed",
            cmd.name,
            "with a generated key in .gyro-recipe/api-key.txt"
            if args.api_key
            else "without API authentication",
        )
    if os.name == "nt":
        run("attrib", "+h", runtime)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--strata-source")
    s.add_argument("--llama-source")
    s.set_defaults(func=sources)
    s = sub.add_parser("install")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--profile-dir", help="Optional separate launcher directory")
    s.add_argument("--models", nargs="+", choices=["M"], default=["M"])
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--workers", type=int)
    s.add_argument("--pack-m")
    auth = s.add_mutually_exclusive_group()
    auth.add_argument(
        "--api-key", action="store_true", help="Opt in to a random local API key"
    )
    auth.add_argument(
        "--no-api-key",
        action="store_true",
        help="Explicitly keep API authentication disabled",
    )
    s.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="Only when the identical file hash was already verified",
    )
    s.set_defaults(func=install)
    add_tuning_arguments(s)
    args = p.parse_args()
    args.func(args)


def add_tuning_arguments(parser):
    parser.add_argument(
        "--pcie-fraction",
        type=float,
        help="Persistent CPU/GPU expert scheduling override, 0 to 1",
    )
    parser.add_argument(
        "--pool-tasks", type=int, help="Persistent task chunk override, 64 to 4096"
    )
    parser.add_argument(
        "--ram-experts-gib",
        type=int,
        help="Expert RAM ceiling, 4 to 16 GiB; still reduced for free RAM",
    )
    parser.add_argument(
        "--vram-reserve-mib",
        type=int,
        help="At least 1024 MiB; increase if the memory guard trips",
    )


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print("Install stopped:", error, file=sys.stderr)
        sys.exit(1)
