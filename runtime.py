"""Shared release provenance and owned-process cleanup; no engine settings changed."""

import contextlib, hashlib, json, os, pathlib, socket, subprocess

ROOT = pathlib.Path(__file__).resolve().parent


def run_batch(script, env=None, check=True, **kwargs):
    """CMD needs an outer quote pair; an env value avoids re-expanding path %."""
    if os.name != "nt":
        raise RuntimeError("Windows command scripts require Windows.")
    environment = dict(os.environ if env is None else env)
    environment["GYRO_BATCH_SCRIPT"] = str(pathlib.Path(script).resolve())
    return subprocess.run(
        'cmd.exe /d /v:off /s /c ""%GYRO_BATCH_SCRIPT%""',
        env=environment,
        check=check,
        **kwargs,
    )


@contextlib.contextmanager
def profile_lock(directory):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".profile.lock").open("a+b")
    locked = False
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(
                "This profile is already running or being installed. Close it before retrying."
            ) from error
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def read_json(path):
    path = pathlib.Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Cannot read "
            + path.name
            + ". Reinstall the profile or restore its matched rollback files."
        ) from error


def write_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def require_free_port(host, port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer from 1 to 65535.")
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError as error:
            raise RuntimeError(
                f"Port {port} is occupied or reserved. Close the previous server or choose another port."
            ) from error


def check_sources(pins):
    for component in ("strata", "llama"):
        root = ROOT / "sources" / component
        pin = pins[component]
        hashes = pin.get("patched_file_sha256", {})
        if not (root / ".git").exists():
            raise RuntimeError(
                "Source checkout is missing. Reinstall from this engine folder."
            )
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True
        ).splitlines()
        extra = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
        if revision != pin["revision"] or (set(changed) | set(extra)) - set(hashes):
            raise RuntimeError(
                "Source checkout changed. Preserve your edits and rebuild in a fresh engine folder."
            )
        for relative, digest in hashes.items():
            path = root / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != digest
            ):
                raise RuntimeError(
                    "Patched source differs: "
                    + relative
                    + ". Reinstall this release in a fresh folder."
                )


def check_build(hardware):
    marker = ROOT / "build-info.json"
    if not marker.is_file():
        raise RuntimeError(
            "Build provenance is missing. Run INSTALL.cmd to build this release."
        )
    build = read_json(marker)
    pins = read_json(ROOT / "pins.json")
    required = {
        "cuda_arch",
        "strata_revision",
        "llama_revision",
        "patch_sha256",
        "binaries",
        "cuda_root",
    }
    if required - set(build):
        raise RuntimeError(
            "Build provenance is incomplete. Rebuild this release with INSTALL.cmd."
        )
    if build["cuda_arch"] != hardware["gpu"]["arch"]:
        raise RuntimeError(
            "GPU architecture changed. Rebuild with INSTALL.cmd --gpu N for the selected card."
        )
    for component in ["strata", "llama"]:
        if build[component + "_revision"] != pins[component]["revision"]:
            raise RuntimeError(
                "Source pin changed. Rebuild this release with INSTALL.cmd."
            )
    if build["patch_sha256"] != pins["strata"]["patch_sha256"]:
        raise RuntimeError("Patch pin changed. Rebuild this release with INSTALL.cmd.")
    if not build.get("binaries"):
        raise RuntimeError(
            "Build provenance contains no binary hashes. Rebuild this release."
        )
    for relative, digest in build["binaries"].items():
        path = ROOT / relative
        if (
            not path.resolve().is_relative_to(ROOT.resolve())
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("Built binary differs from its provenance: " + relative)
    if "patched_file_sha256" in pins["strata"]:
        if "build-engine/strata.exe" not in build["binaries"]:
            raise RuntimeError(
                "Engine binary is missing from build provenance. Rebuild this release."
            )
        check_sources(pins)
    return build


def check_config(cfg):
    if (
        pathlib.Path(cfg["exe"]).resolve()
        != (ROOT / "build-engine/strata.exe").resolve()
    ):
        raise RuntimeError(
            "Launcher points to another engine. Run INSTALL.cmd from the desired engine folder."
        )
    if pathlib.Path(cfg["cwd"]).resolve() != (ROOT / "sources/strata").resolve():
        raise RuntimeError(
            "Launcher points to another source folder. Reinstall it after moving the engine folder."
        )
    args = cfg["args"]
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("Profile args must be a list of strings.")

    def argument(flag):
        if args.count(flag) != 1:
            raise ValueError(
                "Profile must contain exactly one " + flag + ". Reinstall it."
            )
        try:
            return args[args.index(flag) + 1]
        except IndexError as error:
            raise ValueError("Profile is missing a value for " + flag + ".") from error

    pins = read_json(ROOT / "pins.json")
    model = pathlib.Path(argument("--native"))
    weight = pins["models"]["files"][0]
    if (
        not model.is_file()
        or model.name != weight["rfilename"]
        or model.stat().st_size != weight["size"]
    ):
        raise RuntimeError(
            "Pinned Gyro-M GGUF is missing, moved or incomplete. Reinstall using its current folder."
        )
    stamp = cfg.get("_recipe_weight_mtime_ns")
    if stamp is not None and model.stat().st_mtime_ns != stamp:
        raise RuntimeError(
            "Model file changed since installation. Reinstall to verify its SHA256."
        )
    pack = pathlib.Path(argument("--pack"))
    for relative in (
        "dense.bin",
        "index.txt",
        "native_experts.txt",
        "tokenizer/vocab.json",
        "tokenizer/chat_template.jinja",
    ):
        if not (pack / relative).is_file() or (pack / relative).stat().st_size == 0:
            raise RuntimeError(
                "Pack file is missing or empty: "
                + relative
                + ". Reinstall the profile."
            )
    for relative, stamp in cfg.get("_recipe_pack_files", {}).items():
        path = pack / relative
        if (
            not path.resolve().is_relative_to(pack.resolve())
            or not path.is_file()
            or path.stat().st_size != stamp["size"]
            or path.stat().st_mtime_ns != stamp["mtime_ns"]
        ):
            raise RuntimeError(
                "Pack changed since installation: "
                + relative
                + ". Reinstall to verify it."
            )
    if pathlib.Path(cfg["tokenizer"]).resolve() != (pack / "tokenizer").resolve():
        raise RuntimeError(
            "Tokenizer and expert pack do not match. Reinstall the profile."
        )
    ranking = pathlib.Path(argument("--expert-profile"))
    expected = pins["expert_profiles"]["M"]
    if (
        not ranking.is_file()
        or ranking.stat().st_size != expected["size"]
        or hashlib.sha256(ranking.read_bytes()).hexdigest() != expected["sha256"]
    ):
        raise RuntimeError(
            "Expert profile changed or is incomplete. Reinstall it to restore the verified seed."
        )
    context = int(argument("--max-context"))
    if not 1 <= context <= 262144:
        raise ValueError("Context must be between 1 and 262144 tokens.")


def engine_environment(values, build):
    env = os.environ.copy()
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Environment keys and values must be JSON strings.")
        if value:
            env[key] = value
        else:
            env.pop(key, None)
    cuda = pathlib.Path(build["cuda_root"])
    if not (cuda / "bin/nvcc.exe").is_file():
        raise RuntimeError(
            "The CUDA toolkit used for this build is missing. Restore it or rebuild with INSTALL.cmd."
        )
    env["CUDA_PATH"] = str(cuda)
    env["PATH"] = (
        str(cuda / "bin/x64")
        + os.pathsep
        + str(cuda / "bin")
        + os.pathsep
        + env.get("PATH", "")
    )
    return env


def stop_owned(proc, seen):
    """Retain process creation identities so exited parents do not leak known encoders."""
    import psutil

    owned = dict(seen)
    try:
        parent = psutil.Process(proc.pid)
        if proc.poll() is None:
            owned[parent.pid] = parent.create_time()
            for child in parent.children(recursive=True):
                owned[child.pid] = child.create_time()
    except psutil.Error:
        pass
    targets = []
    for pid, created in owned.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                process.terminate()
                targets.append(process)
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(targets, timeout=8)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=8)
    except Exception:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    import os, shutil, subprocess
    from hardware import gpu_info

    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8-sig"))
    binaries = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ["build-engine/strata.exe"]
    }
    nvcc = pathlib.Path(
        shutil.which("nvcc") or pathlib.Path(os.environ["CUDA_PATH"]) / "bin/nvcc.exe"
    )
    value = dict(
        cuda_arch=gpu_info()["arch"],
        strata_revision=pins["strata"]["revision"],
        llama_revision=pins["llama"]["revision"],
        patch_sha256=pins["strata"]["patch_sha256"],
        binaries=binaries,
        compiler_environment=os.environ.get("VSCMD_VER"),
        cuda_root=str(nvcc.parents[1]),
        cuda_version=subprocess.check_output(
            [str(nvcc), "--version"], text=True
        ).strip(),
    )
    (ROOT / "build-info.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print("Build provenance saved for the selected GPU and exact source pins.")
