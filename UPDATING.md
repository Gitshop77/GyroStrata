# Updates and rollback

## Users

Install a new release into a **new permanent engine folder**, keeping the old one. Reuse the same verified GGUF if the model pin is unchanged. Saved packs are reused only when their format and provenance pass validation.

Installation keeps matched backups under .gyro-recipe/rollback/<timestamp>/ beside the launcher. To roll back, stop the server and restore the matching profile JSON, environment JSON and CMD from one backup. Keep the engine/source/build folder those files reference. Do not mix backup versions. Moving the model/engine folder or changing GPU architecture requires reinstalling.

Installs are keyless by default; --api-key opts in. A keyless install removes saved keys from Gyro-M configuration copies, without touching pack/tokenizer data.

## Maintainers

Run check-upstream.py to compare pins with upstream HEAD without changing source. Review already-landed fixes before carrying a patch forward.

Port needed changes in a separate checkout, update source/dependency/model pins, regenerate the complete patch and changed-file hashes, and replay it against pristine source. Rebuild and run installer, resource, CPU/GPU quant, KV, scheduler and context/cache regressions. Compare old/new workloads serially and keep repeatable improvements.

Update README results only from completed measurements. Preserve methodology and distinguish source changes from configuration changes. Refresh verification.json and SHA256SUMS.txt, run verify.py, and scan the tracked tree for private data and generated files before publishing.

Commit the actual source-package files. Sources, builds, models, keys and private benchmark outputs stay local. A ZIP is optional; GitHub source archives already contain the committed files.

AMD installations: rebuild using the same `INSTALL.cmd --backend hip --hip-root "SDK path" --model-dir "model path"` command. HIP keeps its own build/provenance; the CUDA build.cmd does not rebuild a HIP installation.
