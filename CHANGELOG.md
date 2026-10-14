# 1.0.0

- Imported native Gyro trellis/Hadamard support into pinned Strata, with CPU/GPU parity checks.
- Added CPU row-construction optimization, SSD/CPU overlap, RAM expert rebalancing, configurable read merging, conversation-tail retention and suffix-cost reprobes.
- Selected a 256K text profile with Q4 KV, 32K GPU pages, adaptive prefill and resource-aware expert budgets.
- Added resumable model download, GPU-specific Windows builds, existing-model reuse, one launcher and keyless API by default.
- Added source/model/pack verification, runtime memory guards, profile locks, rollback and owned-process cleanup.
- Tested installer failures and path handling; added Windows CI, the three-run speed grid and a hardware-specific AI tuning prompt.

See [README.md](README.md#changelog) for the short feature list and [PATCHES.md](PATCHES.md) for attribution and source changes. This is a tuned Strata baseline, not a rewrite or a universal speed guarantee.

## Unreleased — AMD support

- Opt-in Windows HIP build and launcher for supported discrete Radeon architectures.
- Separate build provenance, HIP device selection and adapter-specific WDDM VRAM monitoring.
- Parity checks before installation; backend-specific defaults and coding-agent instructions.
- Experimental: AMD hardware inference and performance remain unvalidated.
