# GyroStrata - QFN for the desperate

Patched [Strata](https://github.com/Niko1221/Strata) for **Gyro-M on Windows: 12 GB VRAM, 32 GB RAM, 256K text context**. Experts run across VRAM, RAM and SSD.

> Speeds depend heavily on your CPU, GPU bandwidth, RAM and SSD. Use this as a baseline; don't expect identical results.

NVIDIA/CUDA is tested. **AMD/HIP on Windows 11 is experimental:** [AMD setup](AMD.md). Requires an AVX2 CPU, 12 GB dedicated VRAM, 32 GB RAM and a local SSD. [Hardware details](HARDWARE.md).

## Install

NVIDIA prerequisites: [Git](https://git-scm.com/downloads/win), [Python 3.12-3.14](https://www.python.org/downloads/windows/), [VS 2022 C++ Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) with C++/Windows SDK, [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-toolkit-archive), and NVIDIA driver R580+.

Allow **120 GiB for weights/pack**, **4 GiB for engine files**, plus prerequisites. Keep the engine folder in a permanent location.

**Double-click:** DOWNLOAD-MODEL.cmd -> choose a folder -> INSTALL.cmd -> Start-Gyro-M-256K.cmd beside the weights.

**PowerShell:** Change the model path:

```powershell
git clone https://github.com/Gitshop77/GyroStrata.git
cd GyroStrata
$ModelDir = "C:\Your Model Folder"
.\DOWNLOAD-MODEL.cmd "$ModelDir"
.\INSTALL.cmd --model-dir "$ModelDir"
& "$ModelDir\Start-Gyro-M-256K.cmd"
```

Already have the [pinned GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/blob/1ada08422e404aca51ab9df665f089f34c419c6b/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf)? Skip DOWNLOAD-MODEL.cmd. Interrupted downloads resume in the same folder. Moving the engine or weights requires reinstalling with the new path.

The installer downloads source, applies patches, builds for your GPU and checks parity. Original weights stay unchanged.

## Run

API: **http://127.0.0.1:8080/v1** | Model: **qwen38-gyro-m** | Stop: **Ctrl+C**.

No API key by default. Binding to **0.0.0.0** allows network access. Optional: install with **--api-key**; the key is saved in **.gyro-recipe\api-key.txt**. **DELETE-API-KEY.cmd** removes it; restart afterward.

## Settings and speeds

- Q4 K/V, full 256K host context, 32K GPU KV page cache.
- Automatic VRAM expert cache, up to 16 GiB RAM experts, remaining experts on SSD.
- Suffix speculation on; MTP and vision off.
- Thinking enabled without a budget; historical thinking stripped.

Needs **14 GiB available RAM** at startup. Runtime guards stop the model above **min(11.5 GiB VRAM, reported VRAM minus 512 MiB)** or below **512 MiB available RAM**. Sampling can miss brief spikes. Q4 KV and BF16 pack conversions have precision tradeoffs.

Three runs on one NVIDIA 12 GB / 32 GB PC: synthetic Python continuation, thinking off, 256 tokens per reply. Prefill is the initial prompt; output rate covers its reply plus three prefix-reusing replies. Other workloads may be slower.

| Context | Prefill tok/s (mean, range) | Output tok/s (mean, range) |
|---|---:|---:|
| 8K | 953 (951-956) | 29.7 (28.0-30.8) |
| 32K | 1,552 (1,542-1,558) | 36.4 (35.7-36.8) |
| 64K | 1,725 (1,725-1,726) | 38.4 (36.9-40.6) |
| 128K | 1,791 (1,787-1,794) | 37.7 (36.9-38.3) |
| 192K | 1,785 (1,782-1,786) | 36.8 (35.6-37.8) |
| Near 256K | 1,755 (1,753-1,758) | 36.6 (35.4-37.8) |

Long-context retrieval, arithmetic and cache-return checks passed. The 32K GPU cache does **not** limit attention to 32K; leave room for output within 256K.

## Changelog

- Experimental Windows AMD/HIP build, device selection, adapter memory guards and parity-gated installation.
- Imported native Gyro trellis/Hadamard support for CPU, GPU and prefill.
- Faster CPU expert kernels, overlapping SSD reads, RAM expert rebalancing and adjustable read merging.
- Conversation checkpoints/tail retention and suffix-cost reprobes.
- Adaptive prefill, expert caching, CPU/PCIe scheduling and RAM pinning.
- Interleaved expert packing, resumable downloads, existing-weight reuse and GPU-specific builds.
- Source/model/pack verification, parity tests, memory guards and Windows CI.
- Profile locks, atomic configs, rollback, process cleanup and optional API keys.

[Patch attribution](PATCHES.md) | [Updates and rollback](UPDATING.md)

## Tune with Claude Code or Codex

Paste this prompt. It follows our tuning workflow; results on other PCs are not guaranteed.

```text
Install and tune https://github.com/Gitshop77/GyroStrata for my Windows PC.
Read README.md, HARDWARE.md and PATCHES.md; include AMD.md when applicable.

Inspect my GPU/VRAM, CPU, RAM speed, SSD performance, PCIe link and toolchain.
Install the matching CUDA or HIP backend. Reuse existing pinned Gyro-M
weights; install missing requirements and run
setup/parity checks. Measure the baseline before changing settings.

Tune workers/affinity, task chunks, RAM expert placement, CPU/GPU scheduling,
VRAM reserve, prefill and SSD read merging using short, matched trials.
Tune for my actual backend; the published NVIDIA speeds are not AMD targets.
Keep stable gains within the memory guards, preserving requested context
and answer quality. Verify representative tasks, recall and cache reuse.

Run the bundled context grid once before and once after tuning. Repeat only
unclear or unstable results. Report prefill/output tok/s, first-token latency
and memory use. Reuse completed checks unless changes affect them.

Edit source only for a measured bottleneck; update patches/pins, rebuild
and check correctness. Preserve weights and rollback. Ask before driver
changes or destructive actions; leave other models untouched.

Leave one tuned launcher beside the weights and a short results table.
Stop test servers. Do not publish changes.
```

## Contributing

If performance is poor, open an issue with:

```text
GPU and VRAM:
CPU:
RAM capacity and speed:
SSD model/type:
Windows, driver and engine version:
Backend and CUDA/ROCm version:
Context and launch settings:
Prefill and output tok/s:
```

Keep private prompts out of logs.
