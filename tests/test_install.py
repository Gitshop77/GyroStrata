"""Failure/retry, source integrity, build provenance and orphan cleanup regressions."""

import argparse, contextlib, hashlib, io, json, pathlib, subprocess, sys, tempfile, unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recipe, runtime


def git(folder, *args):
    return subprocess.run(
        ["git", "-C", str(folder), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class ReleaseSafety(unittest.TestCase):
    def test_interrupted_clone_retries_and_modified_sources_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            base = root / "base"
            base.mkdir()
            git(base, "init")
            git(base, "config", "core.autocrlf", "false")
            (base / "source.txt").write_bytes(b"base\n")
            git(base, "add", "source.txt")
            git(
                base,
                "-c",
                "user.name=Recipe Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "fixture",
            )
            revision = git(base, "rev-parse", "HEAD")
            (base / "source.txt").write_bytes(b"patched\n")
            diff = subprocess.check_output(["git", "-C", str(base), "diff", "--binary"])
            (root / "patches").mkdir()
            (root / "patches/strata.patch").write_bytes(diff)
            pin = dict(
                url=str(base),
                revision=revision,
                patch_sha256=hashlib.sha256(diff).hexdigest(),
                patched_file_sha256={
                    "source.txt": hashlib.sha256(b"patched\n").hexdigest()
                },
            )
            pins = dict(strata=pin, llama=dict(url=str(base), revision=revision))
            args = argparse.Namespace(strata_source=None, llama_source=None)
            dst = root / "sources/strata"
            dst.mkdir(parents=True)
            git(dst, "init")
            with (
                patch.object(recipe, "HERE", root),
                patch.object(recipe, "PINS", pins),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                recipe.sources(args)
                recipe.sources(args)
                self.assertEqual((dst / "source.txt").read_bytes(), b"patched\n")
                dependency = root / "sources/llama/source.txt"
                dependency.write_bytes(b"personal changes\n")
                with self.assertRaises(RuntimeError):
                    recipe.sources(args)
                self.assertEqual(dependency.read_bytes(), b"personal changes\n")
                dependency.write_bytes(b"base\n")
                (dst / "source.txt").unlink()
                with self.assertRaises(RuntimeError):
                    recipe.sources(args)
                self.assertFalse((dst / "source.txt").exists())

    def test_build_provenance_rejects_changed_gpu_or_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "engine.exe").write_bytes(b"fixture")
            pins = dict(
                strata=dict(revision="s", patch_sha256="p"), llama=dict(revision="l")
            )
            (root / "pins.json").write_text(json.dumps(pins))
            (root / "build-info.json").write_text(
                json.dumps(
                    dict(
                        cuda_arch="89",
                        strata_revision="s",
                        llama_revision="l",
                        patch_sha256="p",
                        cuda_root="fixture",
                        binaries={"engine.exe": hashlib.sha256(b"fixture").hexdigest()},
                    )
                )
            )
            with patch.object(runtime, "ROOT", root):
                runtime.check_build(dict(gpu=dict(arch="89")))
                with self.assertRaises(RuntimeError):
                    runtime.check_build(dict(gpu=dict(arch="86")))
                (root / "engine.exe").write_bytes(b"changed")
                with self.assertRaises(RuntimeError):
                    runtime.check_build(dict(gpu=dict(arch="89")))

    def test_known_child_stopped_after_parent_exit(self):
        import psutil

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);print(p.pid,flush=True)',
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        child = None
        try:
            child = psutil.Process(int(proc.stdout.readline()))
            created = child.create_time()
            proc.wait(timeout=10)
            runtime.stop_owned(proc, {child.pid: created})
            self.assertFalse(child.is_running())
        finally:
            if child is not None:
                try:
                    child.kill()
                except psutil.Error:
                    pass
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
