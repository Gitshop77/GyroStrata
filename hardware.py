"""Resource-aware defaults for Windows CUDA and experimental HIP backends."""

import argparse, copy, csv, json, math, os, re, subprocess

GPU_LIMIT_MIB = 11776
RAM_FLOOR_MIB = 512
MIN_DRIVER_MAJOR = 580


def gpu_info(index=None, backend=None):
    if backend == 'hip':
        from amd import gpu_info as hip_gpu_info
        return hip_gpu_info(0 if index is None else index)
    try:
        index = int(os.environ.get("GYRO_GPU_ID", "0")) if index is None else int(index)
        if index < 0:
            raise ValueError("GPU index must be zero or positive.")
        fields = "index,name,memory.total,memory.used,compute_cap,uuid,driver_version"
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        rows = list(csv.reader(output.strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 7:
            raise ValueError("Incomplete GPU telemetry.")
        row = [value.strip() for value in rows[0]]
        if not re.fullmatch(r"\d+\.\d", row[4]) or not row[5].startswith("GPU-"):
            raise ValueError("GPU architecture or UUID is unavailable.")
        return dict(
            index=int(row[0]),
            name=row[1],
            total_mib=int(row[2]),
            used_mib=int(row[3]),
            arch=row[4].replace(".", ""),
            uuid=row[5],
            driver=row[6],
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot inspect the selected NVIDIA GPU. Check the driver, nvidia-smi and --gpu index. "
            + str(error)
        ) from error


def resource_info(index=None, backend=None, *, detected_gpu=None):
    if os.name != "nt":
        raise RuntimeError("This release requires 64-bit Windows.")
    import ctypes

    try:
        import psutil

        mem = psutil.virtual_memory()
        total, available = mem.total, mem.available
        cores = psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 2) // 2)
    except ImportError:
        # Bootstrap has no third-party packages yet; Windows provides RAM totals.
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (x, ctypes.c_ulonglong)
                for x in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        mem = Status()
        mem.length = ctypes.sizeof(mem)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            raise OSError("Unable to inspect available RAM")
        total, available = mem.total, mem.available
        cores = max(1, (os.cpu_count() or 2) // 2)
    avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    return dict(
        gpu=(detected_gpu if detected_gpu is not None else
             gpu_info(index, backend) if backend else gpu_info(index)),
        physical_cores=cores,
        avx2=avx2,
        ram_gib=total / 2**30,
        available_ram_gib=available / 2**30,
    )


def adapt(profile, environment, hardware, workers=None, tuning=None):
    cfg = copy.deepcopy(profile)
    env = dict(environment)
    gpu = hardware["gpu"]
    backend = gpu.get('backend', 'cuda')
    tuning = dict(tuning or {})
    unknown = set(tuning) - {
        "pcie_fraction",
        "pool_tasks",
        "ram_experts_gib",
        "vram_reserve_mib",
    }
    if unknown:
        raise ValueError("Unknown tuning overrides: " + ", ".join(sorted(unknown)))
    if workers is not None and (type(workers) is not int or not 0 <= workers <= 1024):
        raise ValueError(
            "Worker override must be an integer from 0 (automatic) to 1024."
        )
    if not hardware.get("avx2", True):
        raise ValueError("This engine requires an AVX2-capable CPU.")
    if backend not in ('cuda', 'hip'):
        raise ValueError('Unknown GPU backend.')
    if backend == 'hip':
        from amd import ARCHS
        if gpu['arch'] not in ARCHS:
            raise ValueError('Unsupported experimental HIP architecture.')
    elif int(gpu["arch"]) < 75:
        raise ValueError(
            "This CUDA engine needs an NVIDIA GPU with compute capability 7.5 or newer."
        )
    if backend == 'cuda' and "driver" in gpu:
        driver = re.fullmatch(r"(\d+)\.\d+(?:\.\d+)?", gpu["driver"])
        if not driver or int(driver[1]) < MIN_DRIVER_MAJOR:
            raise ValueError(
                "CUDA 13 requires an NVIDIA R580 or newer driver supported by your GPU."
            )
    if gpu["total_mib"] < 11500:
        raise ValueError(
            "This full-context engine needs a 12 GiB or larger discrete GPU."
        )
    if hardware["ram_gib"] < 30:
        raise ValueError("This full-context engine needs 32 GB or more system RAM.")
    limit = min(GPU_LIMIT_MIB, gpu["total_mib"] - 512)
    if gpu["used_mib"] >= limit:
        raise ValueError(
            "The selected GPU is already at the memory limit. Close the other model or GPU-heavy applications first."
        )
    # Leave room for full host KV, a near-full conversation snapshot, Python and OS.
    # The engine independently rebalances expert RAM and sizes VRAM after its weights load.
    experts = min(16, max(0, math.floor(hardware["available_ram_gib"] - 10)))
    if backend == 'hip':
        experts = min(experts, 13)
    if experts < 4:
        raise ValueError(
            "Close memory-heavy applications first: at least 14 GiB available RAM is needed."
        )
    if "ram_experts_gib" in tuning:
        budget = tuning["ram_experts_gib"]
        if type(budget) is not int or not 4 <= budget <= 16:
            raise ValueError("Expert RAM override must be an integer from 4 to 16 GiB.")
        experts = min(experts, budget)

    def set_arg(flag, value):
        try:
            cfg["args"][cfg["args"].index(flag) + 1] = str(value)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError(
                "Profile is missing a value for " + flag + ". Reinstall it."
            ) from error

    set_arg("--resident-budget-gib", experts)
    set_arg("--expert-cache", "auto")
    set_arg("--pool-workers", 0 if workers is None else workers)
    core_workers = (
        max(1, hardware["physical_cores"] - 1) if workers in (None, 0) else workers
    )
    tasks = min(192, max(64, math.ceil(core_workers * (10) / 64) * 64))
    if "pool_tasks" in tuning:
        tasks = tuning["pool_tasks"]
        if type(tasks) is not int or not 64 <= tasks <= 4096:
            raise ValueError("Task override must be an integer from 64 to 4096.")
    set_arg("--pool-tasks", tasks)
    set_arg("--conversation-cache-min-free-mib", 1024 if backend == 'hip' else RAM_FLOOR_MIB)
    fraction = tuning.get("pcie_fraction", 0.75)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("PCIe fraction must be a finite number from 0 to 1.")
    if "--pcie-frac" not in cfg["args"]:
        cfg["args"] += ["--pcie-frac", str(fraction)]
    else:
        set_arg("--pcie-frac", fraction)
    if backend == 'hip' and 'pcie_fraction' not in tuning:
        at = cfg['args'].index('--pcie-frac')
        del cfg['args'][at:at+2]
        fraction = 'auto'
    base_reserve = tuning.get("vram_reserve_mib", 2048 if backend == 'hip' else 1024)
    if type(base_reserve) is not int or not 1024 <= base_reserve <= 8192:
        raise ValueError("VRAM reserve must be an integer from 1024 to 8192 MiB.")
    # Larger GPUs may use the engine, but its absolute 11.5 GiB GPU cap stays fixed.
    set_arg("--vram-reserve-mib", base_reserve + max(0, gpu["total_mib"] - 12288))
    env.update(
        STRATA_RAM_REBALANCE="512",
        STRATA_PARTIAL_PIN_GIB=str(min(8, max(2, experts // 2))),
        CUDA_VISIBLE_DEVICES=gpu["uuid"],
    )
    if backend == 'hip':
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.update(HIP_VISIBLE_DEVICES=str(gpu['index']), ROCR_VISIBLE_DEVICES='',
                   STRATA_PREFILL_CPU_SHARE='0', STRATA_PARTIAL_PIN='0')
    selection = dict(
        gpu_name=gpu["name"],
        cuda_arch=gpu["arch"],
        physical_cores=hardware["physical_cores"],
        ram_experts_gib=experts,
        partial_pin_gib=int(env["STRATA_PARTIAL_PIN_GIB"]),
        gpu_limit_mib=limit,
        ram_floor_mib=RAM_FLOOR_MIB,
        pool_tasks=tasks,
        pcie_fraction=fraction,
        vram_reserve_mib=base_reserve + max(0, gpu["total_mib"] - 12288),
        tuning_overrides=tuning,
        backend=backend,
    )
    if backend == 'hip':
        selection['hip_arch'] = selection.pop('cuda_arch')
    return cfg, env, selection


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["arch", "inspect"])
    p.add_argument("--gpu", type=int)
    p.add_argument('--backend', choices=['cuda', 'hip'], default='cuda')
    a = p.parse_args()
    if a.action == "arch":
        g = gpu_info(a.gpu, a.backend)
        if a.backend == 'cuda' and int(g["arch"]) < 75:
            raise SystemExit("GPU compute capability below the supported 7.5 floor")
        print(g["arch"])
    else:
        print(json.dumps(resource_info(a.gpu, a.backend), indent=2))
