"""One entry point: check prerequisites, fetch pinned sources, build and install profiles."""

import argparse, json, os, pathlib, re, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent


def select_model_directory(explicit, saved_file):
    filename = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
    folder = explicit
    if not folder and saved_file.is_file():
        saved = saved_file.read_text(encoding="utf-8-sig").strip()
        if saved and (pathlib.Path(saved) / filename).is_file():
            folder = saved
            print(
                "Using the model folder selected by DOWNLOAD-MODEL.cmd:",
                folder,
                flush=True,
            )
    if not folder:
        folder = (
            input("Folder containing your Gyro-M GGUF weights: ").strip().strip('"')
        )
    if not folder or not str(folder).strip():
        raise RuntimeError(
            'No model folder selected. Run INSTALL.cmd --model-dir "C:\\Your Model Folder".'
        )
    return pathlib.Path(folder).resolve()


def delete_api_key(profiledir):
    runtime = pathlib.Path(profiledir) / ".gyro-recipe"
    if not (runtime / "M.json").is_file():
        raise RuntimeError("No installed Gyro-M profile was found in " + str(runtime))
    from recipe import configure_api_key
    from runtime import profile_lock

    with profile_lock(runtime):
        configure_api_key(runtime, True)


def vcvars():
    if os.environ.get("GYRO_VCVARS"):
        override = pathlib.Path(os.environ["GYRO_VCVARS"])
        if not override.is_file():
            raise RuntimeError(
                "GYRO_VCVARS does not point to an existing vcvars64.bat."
            )
        return str(override.resolve())
    if shutil.which("cl") and os.environ.get("VSCMD_VER", "").startswith("17."):
        return ""
    locator = (
        pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not locator.is_file():
        raise RuntimeError(
            "Install VS 2022 C++ Build Tools (Desktop development with C++), then rerun INSTALL.cmd."
        )
    home = subprocess.check_output(
        [
            str(locator),
            "-latest",
            "-products",
            "*",
            "-version",
            "[17.0,18.0)",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    result = pathlib.Path(home) / "VC/Auxiliary/Build/vcvars64.bat"
    if not home or not result.is_file():
        raise RuntimeError("VS 2022 C++ x64 tools were not found.")
    return str(result)


def main():
    from recipe import add_tuning_arguments

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["M"],
        help="Omit to install the model files present in the folder",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument('--backend', choices=['cuda', 'hip'], default='cuda')
    p.add_argument('--hip-root', help='ROCm development SDK root for --backend hip')
    p.add_argument("--workers", type=int)
    p.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites without building/installing",
    )
    p.add_argument("--profile-dir")
    p.add_argument("--pack-m")
    auth = p.add_mutually_exclusive_group()
    auth.add_argument(
        "--api-key", action="store_true", help="Opt in to a random local API key"
    )
    auth.add_argument(
        "--no-api-key",
        action="store_true",
        help="Explicitly keep API authentication disabled",
    )
    p.add_argument(
        "--delete-api-key",
        action="store_true",
        help="Remove API authentication from an existing installation without rebuilding",
    )
    p.add_argument("--skip-weight-hash", action="store_true")
    add_tuning_arguments(p)
    a = p.parse_args()
    if os.name != "nt":
        raise RuntimeError(
            "This installer requires 64-bit Windows."
        )
    if a.delete_api_key:
        if a.api_key:
            raise RuntimeError("Choose either --delete-api-key or --api-key.")
        saved_dir = ROOT / ".model-dir"
        modeldir = select_model_directory(a.model_dir, saved_dir)
        profiledir = (
            pathlib.Path(a.profile_dir).resolve() if a.profile_dir else modeldir
        )
        delete_api_key(profiledir)
        print(
            "Deleted saved API keys. The existing launcher now runs without API authentication; restart it if the server is open.",
            flush=True,
        )
        return
    from verify import verify_release

    verify_release(ROOT)
    if not (3, 12) <= sys.version_info[:2] <= (3, 14):
        raise RuntimeError("Use standalone Python 3.12-3.14.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Use 64-bit Python for these large GGUF files.")
    if not shutil.which("git"):
        raise RuntimeError("Install Git for Windows, then rerun INSTALL.cmd.")
    if a.backend == 'hip':
        from amd_setup import install as install_hip
        return install_hip(a)
    from hardware import resource_info, adapt

    hardware = resource_info(a.gpu)
    gpu = hardware["gpu"]
    if int(gpu["arch"]) < 75 or gpu["total_mib"] < 11500:
        raise RuntimeError(
            "Select a supported 12 GiB or larger NVIDIA GPU with --gpu N."
        )
    tuning = {
        key: getattr(a, key)
        for key in [
            "pcie_fraction",
            "pool_tasks",
            "ram_experts_gib",
            "vram_reserve_mib",
        ]
        if getattr(a, key) is not None
    }
    adapt(
        json.loads((ROOT / "profiles/M.json").read_text(encoding="utf-8-sig")),
        {},
        hardware,
        a.workers,
        tuning,
    )
    cuda = pathlib.Path(os.environ.get("CUDA_PATH", ""))
    nvcc = (
        str(cuda / "bin/nvcc.exe")
        if os.environ.get("CUDA_PATH")
        else shutil.which("nvcc")
    )
    if not nvcc:
        raise RuntimeError(
            "Install CUDA Toolkit 13.2 or set CUDA_PATH to that toolkit."
        )
    if not pathlib.Path(nvcc).is_file():
        raise RuntimeError(
            "Install CUDA Toolkit 13.2; the existing driver is not changed by this installer."
        )
    version = re.search(
        r"release (\d+)\.(\d+)", subprocess.check_output([nvcc, "--version"], text=True)
    )
    if not version or (int(version[1]), int(version[2])) != (13, 2):
        raise RuntimeError(
            "This release was validated with CUDA Toolkit 13.2. Select that toolkit; other toolkits need separate build validation."
        )
    supported = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    if "sm_" + gpu["arch"] not in supported:
        raise RuntimeError(
            "The selected CUDA toolkit cannot compile for this GPU architecture."
        )
    compiler = vcvars()
    print(
        "Prerequisites found:",
        gpu["name"],
        "; CUDA architecture",
        gpu["arch"],
        flush=True,
    )
    if a.check:
        return
    from runtime import require_free_port, run_batch

    require_free_port("0.0.0.0", 8080)
    saved_dir = ROOT / ".model-dir"
    modeldir = select_model_directory(a.model_dir, saved_dir)
    filenames = {"M": "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"}
    if a.models is None:
        a.models = [m for m in ["M"] if (modeldir / filenames[m]).is_file()]
    if not a.models or not all((modeldir / filenames[m]).is_file() for m in a.models):
        raise RuntimeError(
            "Selected GGUF files were not found; download the pinned files linked in README.md first."
        )
    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8-sig"))
    selected = [filenames[m] for m in a.models]
    verified = {}
    from recipe import sha

    for filename in selected:
        pin = next(x for x in pins["models"]["files"] if x["rfilename"] == filename)
        path = modeldir / filename
        if path.stat().st_size != pin["size"]:
            raise RuntimeError(
                "Wrong or incomplete file size: "
                + filename
                + "; use the pinned model revision."
            )
        if not a.skip_weight_hash:
            print(
                "Verifying model SHA256 before building; this can take a few minutes...",
                flush=True,
            )
            if sha(path) != pin["lfs"]["sha256"]:
                raise RuntimeError(
                    "Wrong model SHA256. The existing file was left untouched; use the pinned model revision."
                )
        verified[filename] = (path.stat().st_size, path.stat().st_mtime_ns)
    if shutil.disk_usage(ROOT).free < 4 * 2**30:
        raise RuntimeError(
            "Leave at least 4 GiB free on the engine drive for sources, packages and build files."
        )
    saved_dir.write_text(str(modeldir), encoding="utf-8")

    def run(*args, **kwargs):
        subprocess.run([str(x) for x in args], check=True, **kwargs)

    run(sys.executable, ROOT / "recipe.py", "sources")
    env = os.environ.copy()
    env.update(GYRO_PYTHON=sys.executable, GYRO_VCVARS=compiler, GYRO_GPU_ID=str(a.gpu))
    env["CUDA_PATH"] = str(pathlib.Path(nvcc).parents[1])
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    env["PATH"] = str(pathlib.Path(nvcc).parent) + os.pathsep + env.get("PATH", "")
    run_batch(ROOT / "build.cmd", cwd=ROOT, env=env)
    for filename, stamp in verified.items():
        path = modeldir / filename
        if (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
            raise RuntimeError(
                "Model file changed during setup. Rerun to verify it before installation."
            )
    command = [
        ROOT / ".venv/Scripts/python.exe",
        ROOT / "recipe.py",
        "install",
        "--model-dir",
        modeldir,
        "--models",
        *a.models,
        "--gpu",
        a.gpu,
    ]
    for name in ["profile_dir", "pack_m"]:
        if getattr(a, name):
            command += ["--" + name.replace("_", "-"), getattr(a, name)]
    command.append("--api-key" if a.api_key else "--no-api-key")
    # Verified above before compiling; unchanged file identity avoids a second full read.
    command.append("--skip-weight-hash")
    for name in [
        "workers",
        "pcie_fraction",
        "pool_tasks",
        "ram_experts_gib",
        "vram_reserve_mib",
    ]:
        if getattr(a, name) is not None:
            command += ["--" + name.replace("_", "-"), str(getattr(a, name))]
    run(*command)
    print(
        "Done. Open a Start-Gyro launcher beside your weights. No model weights or drivers were modified.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        KeyError,
        subprocess.SubprocessError,
    ) as e:
        print("Setup stopped:", e, file=sys.stderr)
        sys.exit(1)
