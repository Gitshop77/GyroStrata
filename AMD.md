# AMD on Windows (experimental)

Builds the same pinned Strata source and Gyro patches with HIP.

Requires Windows 11 x64, an AVX2 CPU, 32 GB RAM, a local SSD, a current AMD driver, VS 2022 C++ Build Tools, Git and Python 3.12–3.14. Install a complete [ROCm development SDK](https://rocm.docs.amd.com/) with the compiler, hipBLAS and AMDGPU bitcode. The upstream Windows builder uses ROCm 10.2 TheRock packages; use a compatible complete SDK.

Candidate discrete architectures: `gfx1030`, `gfx1031`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1200`, `gfx1201`, with at least 12 GB dedicated VRAM. Integrated GPUs and mixed-vendor execution are excluded. Your card and driver must pass the installer checks.

## Install

Run **DOWNLOAD-MODEL.cmd**, or reuse the pinned weights. Then run in PowerShell:

```powershell
$ModelDir = "C:\Your Model Folder"
$HipRoot = "C:\Your ROCm SDK"
.\INSTALL.cmd --backend hip --hip-root "$HipRoot" --model-dir "$ModelDir"
& "$ModelDir\Start-Gyro-M-256K.cmd"
```

If an integrated GPU is HIP device 0, add `--gpu 1` to select the discrete card. Installation runs GPU/CPU quantization parity and cache checks before creating the launcher. A failed check stops installation.

CUDA and HIP keep separate builds. Device selection uses HIP numbering; the VRAM guard monitors that adapter through its Windows LUID. Missing telemetry stops the model.

Defaults: up to 13 GiB RAM experts, a 2048 MiB VRAM reserve, automatic PCIe scheduling, 256K context and Q4 KV. MTP and vision are off. Repeat the same install command to rebuild.

**AMD inference and performance have not been validated on hardware for this project.** The README speeds are NVIDIA results. This is an experimental build path with local correctness gates, not a promise of compatibility with every Radeon. Report your GPU, driver, parity results and speeds.

[Upstream AMD implementation](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md).
