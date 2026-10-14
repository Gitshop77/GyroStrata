# Engine changes and attribution

GyroStrata reconstructs [official Strata](https://github.com/Niko1221/Strata) at a0116b2782a5b82622358c04a06496485d12b2d6, then applies patches/strata.patch. The full diff changes **47 files** (including tests): **3,583 added lines and 75 removed lines**, plus a binary Git hunk. Every resulting changed file is hashed in pins.json.

## Imported compatibility

The Gyro author's [Strata integration](https://github.com/agentionai/Strata), including the rc2 work at 9ff02c9fa9c86a88088370925b3c9131cc5d6c26, supplied native trellis quantization and APR/Hadamard support. CPU, GPU, prefill and packing paths must honor those rotations. This is existing public work, not a new quantization algorithm authored here.

The unchanged [agentionai llama.cpp](https://github.com/agentionai/llama.cpp) dependency is pinned at dc255f2a4b6e4211b555b0c05260b2523281f314. It defines the custom tensor types the GGUF needs. Ordinary llama.cpp is not interchangeable.

## Local source changes

- **CPU trellis path:** retained the measured row-construction optimization, with runtime ISA dispatch and numerical parity checks.
- **SSD/CPU overlap:** a persistent async worker fetches file experts while resident experts compute. STRATA_RESIDENT_IO_OVERLAP=1 enables it.
- **RAM placement:** recently used disk experts can replace cold RAM experts after a synchronized reply, without increasing RAM capacity. STRATA_RAM_REBALANCE=512 bounds swaps per reply.
- **Read merging:** STRATA_IO_MERGE_GAP_KIB exposes the read-gap tradeoff; the selected default remains 1024 KiB.
- **Conversation caching:** explicit tail retention and a near-tail checkpoint reduce rereading when returning to a long prompt. Memory pressure can still evict checkpoints.
- **Suffix policy:** STRATA_LOOKUP_REPROBE_ROUNDS=128 refreshes stale verification costs after expert placement warms; a regression covers an initially unprofitable window.
- **Scheduling/build:** per-request pool-task tuning and the reviewed MSVC WAITPKG guard. UMWAIT stays off.

Supporting readers, packing changes and regression tests are part of the diff. The binary hunk is intentional: the pinned original expert_source.cpp contained a literal NUL. Do not reformat the patch.

## Configuration and tools

Adaptive prefill, the 95% prefill VRAM loan, automatic expert caching, Q4 KV, the 32K GPU KV page cache, RAM budgets, partial pinning and CPU/PCIe settings are configuration choices. Some are inherited Strata features; they are not all new engine code.

The interleaved experts.bin groups unchanged native expert bytes for SSD reads. Packing also derives required BF16 projections and records rounding in conversions.json. That conversion is not lossless; the original GGUF, native experts and disk-backed PLE weights are never rewritten.

Installation/launch tools add source and binary checks, model/pack validation, capacity checks, profile locks, rollback, safe key removal and process cleanup. They support one Gyro-M text profile. A full standalone Strata fork is optional: the pinned source plus patch reconstructs this engine reproducibly.

The 8192 prefill experiment changed chunk sizing; it was not an expert-cache size change or a whole-engine redesign.
