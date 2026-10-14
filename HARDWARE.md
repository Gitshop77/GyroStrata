# Hardware and tuning

This is a Windows x64 baseline for **12 GB or larger NVIDIA VRAM and 32 GB or more RAM**, an **AVX2 CPU**, and a **local SSD**. Capacity alone does not guarantee compatibility or performance.

The engine builds for the selected GPU's actual CUDA architecture. Examples of eligible 12 GB variants include RTX 3060/3080/3080 Ti (8.6), RTX 4070/4070 SUPER/4070 Ti (8.9), and RTX 5070 (12.0). Check the exact SKU: laptop and desktop cards can have different VRAM. Architecture eligibility comes from [NVIDIA's compute-capability table](https://developer.nvidia.com/cuda/gpus); it is not evidence that we benchmarked each card.

CUDA Toolkit 13.2 and VS 2022 C++ x64 tools plus a Windows SDK are the validated build combination. CUDA 13.x has an [R580 driver floor](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html); some newer features can require a newer driver. Numerical tests must pass on the installed driver. The [CUDA Windows guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-microsoft-windows/index.html) covers prerequisites. This installer never changes drivers. Pascal/Volta and GPUs below compute capability 7.5 are outside this release.

## How the baseline adapts

| Resource | Policy | What still affects speed |
|---|---|---|
| GPU | Compile for its detected architecture; select by UUID. Rebuild for a different architecture. | Core throughput, VRAM bandwidth, power limits and background apps. |
| VRAM | Automatic expert cache; minimum 1024 MiB reserve. Guard is min(11776 MiB, reported total minus 512 MiB). | Windows can report less memory than the card's advertised capacity. A sampled guard can briefly overshoot before stopping. |
| CPU | Native automatic workers; task chunks scaled from physical cores, 64–192 by default. AVX2 checked. | Core speed, hybrid cores, SIMD, affinity and contention. |
| RAM | Expert budget = min(16 GiB, floor(initial available GiB minus 10)). At least 14 GiB initially free; runtime floor 512 MiB. | Channels, speed, latency and other applications. |
| PCIe | Default scheduling fraction 0.75, with a persistent override. | Link width/generation and actual transfer bandwidth are not automatically calibrated. |
| SSD | Read-only inference data access, overlapping expert reads, interleaved expert layout. | NVMe/SATA, random latency, throttling and filesystem. Installation writes the download and pack once; the OS can still write logs or page data. |
| Context | Full 262144 host context with 32768 GPU KV pages and Q4 K/V. | KV transfers, output length and workload; quantization has a precision tradeoff. |

Larger GPUs retain the absolute 11.5 GiB guard. More RAM does not automatically raise the 16 GiB expert ceiling. This baseline does not apply driver or clock tuning.

Fallback partial pinning scales from 2 to 8 GiB. Upstream can pin the entire resident expert allocation when that succeeds; this setting is not a hard cap on all page-locked memory.

## Tune the host

Use the README's Claude Code/Codex prompt, or compare these controls on identical prompts and background load:

- **--workers N:** zero is native automatic selection.
- **--pool-tasks N:** 64–4096 task chunks.
- **--pcie-fraction X:** 0–1.
- **--ram-experts-gib N:** 4–16 GiB; free-RAM safety still reduces it.
- **--vram-reserve-mib N:** 1024–8192 MiB; increase if the GPU guard trips.
- **--gpu N:** select the physical NVIDIA GPU; rebuild when its architecture differs.

Pass controls to INSTALL.cmd with --model-dir. To change settings without rebuilding, run:

```powershell
.\.venv\Scripts\python.exe recipe.py install --model-dir "C:\Your Model Folder" --workers 8 --pool-tasks 128 --pcie-fraction 0.5
```

That is an experiment, not a recommended replacement. Omit overrides to restore defaults. The launcher preserves chosen overrides while checking current free RAM.

CUDA_PATH selects Toolkit 13.2 when multiple toolkits are installed. For portable VS tools, set GYRO_VCVARS to their vcvars64.bat; do not disable compiler checks.

**--pack-m** accepts an independently trusted compatible pack. Native layout, Hadamard metadata, tokenizer and expert-sidecar provenance are checked, but those checks do not cryptographically prove every externally supplied dense/expert byte. Fresh installation generates its own pack. Only use --skip-weight-hash if the identical file has already been verified.

## Verification

Resource tests simulate architectures, capacity boundaries and core counts. They cannot substitute for physical performance tests on other PCs. Installation runs CPU/Hadamard/KV/scheduler tests and a real-model CPU/GPU expert parity check; launch verifies source/binary identity and detects changed model/pack files.

The included benchmark is serial and writes private results under local-results:

```powershell
.\.venv\Scripts\python.exe benchmarks\context-grid.py --model-dir "C:\Your Model Folder" --out local-results\run1
```

Use run2 and run3 for independent repetitions. Keep only repeatable gains; check representative tasks, full context, reasoning and cache return as well as tok/s. Generated profiles, results and logs are excluded from Git by default.
