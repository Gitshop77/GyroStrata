"""User mistakes, release integrity, capacity boundaries and process isolation."""

import copy
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hardware
import recipe
import runtime
import setup as setup_entry
from verify import verify_release


class UserFailures(unittest.TestCase):
    def test_empty_model_selection_stops(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("builtins.input", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "No model folder"):
                setup_entry.select_model_directory(
                    None, pathlib.Path(temp) / ".model-dir"
                )

    def test_key_removal_never_changes_pack_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            vocab = root / "pack-M" / "tokenizer" / "vocab.json"
            vocab.parent.mkdir(parents=True)
            vocab.write_text('{"api_key": 123, "token": 456}', encoding="utf-8")
            before = vocab.read_bytes()
            (root / "M.json").write_text('{"api_key": "fixture", "port": 8080}')
            recipe.configure_api_key(root, True)
            self.assertEqual(vocab.read_bytes(), before)
            self.assertNotIn("api_key", json.loads((root / "M.json").read_text()))

    def test_gpu_telemetry_validation(self):
        good = "0, Example GPU, 11911, 500, 8.6, GPU-TEST, 616.92\n"
        with patch("subprocess.check_output", return_value=good):
            self.assertEqual(hardware.gpu_info()["arch"], "86")
        for bad in [
            "",
            good + good,
            good.replace("8.6", "N/A"),
            good.replace("11911", "N/A"),
        ]:
            with patch("subprocess.check_output", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
                    hardware.gpu_info()
        with self.assertRaises(RuntimeError):
            hardware.gpu_info(-1)

    def test_architectures_core_counts_and_ram_boundaries(self):
        profile = json.loads((ROOT / "profiles/M.json").read_text())
        baseline = dict(
            gpu=dict(
                name="Example",
                total_mib=11911,
                used_mib=500,
                arch="86",
                uuid="GPU-TEST",
                driver="616.92",
            ),
            avx2=True,
            physical_cores=6,
            ram_gib=31.9,
            available_ram_gib=24,
        )
        for arch in ["75", "80", "86", "89", "120", "121"]:
            for cores in [2, 4, 6, 8, 16, 32, 64]:
                host = copy.deepcopy(baseline)
                host["gpu"]["arch"], host["physical_cores"] = arch, cores
                _, _, choice = hardware.adapt(profile, {}, host)
                self.assertEqual(choice["gpu_limit_mib"], 11399)
                self.assertEqual(choice["ram_experts_gib"], 14)
                self.assertTrue(64 <= choice["pool_tasks"] <= 192)
        for key, value in [
            ("avx2", False),
            ("available_ram_gib", 13.99),
            ("ram_gib", 16),
        ]:
            host = copy.deepcopy(baseline)
            host[key] = value
            with self.assertRaises(ValueError):
                hardware.adapt(profile, {}, host)
        for key, value in [
            ("arch", "61"),
            ("arch", "70"),
            ("driver", "579.99"),
            ("total_mib", 8192),
            ("used_mib", 11400),
        ]:
            host = copy.deepcopy(baseline)
            host["gpu"][key] = value
            with self.assertRaises(ValueError):
                hardware.adapt(profile, {}, host)
        for workers in [-1, 1.5, True, 1025]:
            with self.assertRaises(ValueError):
                hardware.adapt(profile, {}, baseline, workers=workers)

    def test_profile_lock_releases_and_rejects_second_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            with runtime.profile_lock(temp):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with runtime.profile_lock(temp):
                        self.fail("second owner acquired the profile")
            with runtime.profile_lock(temp):
                pass

    def test_occupied_port_and_invalid_port(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            with self.assertRaisesRegex(RuntimeError, "occupied or reserved"):
                runtime.require_free_port("127.0.0.1", occupied.getsockname()[1])
        for port in [0, -1, 65536, True, "8080"]:
            with self.assertRaises(ValueError):
                runtime.require_free_port("127.0.0.1", port)

    def test_manifest_rejects_changed_empty_duplicate_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            target = root / "file.txt"
            target.write_bytes(b"correct")
            digest = hashlib.sha256(b"correct").hexdigest()
            manifest = root / "SHA256SUMS.txt"
            line = digest + "  file.txt\n"
            manifest.write_text(line)
            self.assertEqual(verify_release(root), 1)
            for bad in [
                "",
                line + line,
                digest + "  ../file.txt\n",
                "wrong  file.txt\n",
            ]:
                manifest.write_text(bad)
                with self.assertRaises((RuntimeError, ValueError)):
                    verify_release(root)
            manifest.write_text(line)
            target.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                verify_release(root)

    @unittest.skipUnless(os.name == "nt", "Windows command wrappers")
    def test_wrapper_failure_codes_and_batch_paths(self):
        with tempfile.TemporaryDirectory(prefix="gyro & ! 50% ") as temp:
            root = pathlib.Path(temp)
            fake = root / "entry.py"
            fake.write_text("import sys; raise SystemExit(7)", encoding="utf-8")
            launcher = root / "Start.cmd"
            launcher.write_text(
                recipe.launcher_text(sys.executable, fake), encoding="utf-8"
            )
            env = dict(os.environ, GYRO_NO_PAUSE="1", GYRO_PYTHON=sys.executable)
            result = runtime.run_batch(
                launcher,
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            (root / "setup.py").write_text(fake.read_text(), encoding="utf-8")
            for name in ["INSTALL.cmd", "DELETE-API-KEY.cmd"]:
                (root / name).write_bytes((ROOT / name).read_bytes())
                result = runtime.run_batch(
                    root / name,
                    env=env,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 7, result.stderr)

    def test_missing_profile_has_clear_error_and_no_traceback(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "launch.py"), "missing-profile.json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Installed profile is missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
