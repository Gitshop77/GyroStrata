"""Dry-run validation and cleanup on telemetry or memory failure."""

import contextlib
import copy
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launch
import runtime
import amd


class LaunchGuard(unittest.TestCase):
    def run_fixture(self, used, available, check=False, backend='cuda'):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            profile = root / "M.json"
            cfg = json.loads((ROOT / "profiles/M.json").read_text())
            cfg["api_key"] = "only-a-test-secret"
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            profile.with_suffix(".env.json").write_text("{}")
            host = dict(
                gpu=dict(
                    name="Example",
                    arch="89",
                    uuid="GPU-TEST",
                    total_mib=12288,
                    used_mib=500,
                ),
                physical_cores=8,
                ram_gib=31.9,
                available_ram_gib=24,
            )
            proc = MagicMock(pid=424242)
            if backend == 'hip':
                cfg['_recipe_backend'] = 'hip'
                cfg['_recipe_gpu_index'] = 1
                profile.write_text(json.dumps(cfg), encoding='utf-8')
                host['gpu'].update(backend='hip',index=1,arch='gfx1101',luid='00000000_0000abcd')
            proc.poll.return_value = None
            process = MagicMock()
            process.create_time.return_value = 123
            process.children.return_value = []
            output = io.StringIO()
            with (
                patch.object(launch, "resource_info", return_value=host),
                patch.object(launch, "check_build", return_value={}),
                patch.object(launch, "check_config"),
                patch.object(launch, "engine_environment", return_value={}),
                patch.object(launch, "require_free_port"),
                patch.object(launch.subprocess, "Popen", return_value=proc) as start,
                patch.object(launch.subprocess, "check_output", return_value=used),
                patch.object(amd, 'memory_used', side_effect=used if isinstance(used, Exception) else None,
                             return_value=used),
                patch.object(launch.psutil, "Process", return_value=process),
                patch.object(
                    launch.psutil,
                    "virtual_memory",
                    return_value=SimpleNamespace(available=available),
                ),
                patch.object(launch, "stop_owned") as stop,
                contextlib.redirect_stdout(output),
            ):
                if check:
                    self.assertEqual(launch.run_profile(profile, True), 0)
                    start.assert_not_called()
                    stop.assert_not_called()
                else:
                    with self.assertRaises(RuntimeError):
                        launch.run_profile(profile)
                    stop.assert_called_once_with(proc, {424242: 123})
                self.assertNotIn("only-a-test-secret", output.getvalue())

    def test_gpu_limit_stops_only_owned_server(self):
        self.run_fixture("11777\n", 2**30)

    def test_ram_floor_stops_only_owned_server(self):
        self.run_fixture("10000\n", 256 * 2**20)

    def test_lost_or_invalid_telemetry_stops_owned_server(self):
        for value in ["N/A\n", "", "bad"]:
            self.run_fixture(value, 2**30)

    def test_check_does_not_start_model(self):
        self.run_fixture("10000\n", 2**30, check=True)

    def test_hip_guard_stops_owned_model_on_limit_or_missing_counter(self):
        self.run_fixture(11777,2**30,backend='hip')
        self.run_fixture(RuntimeError('Missing adapter counter'),2**30,backend='hip')


class ProfileIdentity(unittest.TestCase):
    def test_stale_paths_weights_pack_and_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "build-engine").mkdir()
            source = root / "sources/strata"
            source.mkdir(parents=True)
            model = root / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
            model.write_bytes(b"model")
            pack = root / "pack"
            (pack / "tokenizer").mkdir(parents=True)
            for relative in [
                "dense.bin",
                "index.txt",
                "native_experts.txt",
                "tokenizer/vocab.json",
                "tokenizer/chat_template.jinja",
            ]:
                (pack / relative).write_bytes(b"fixture")
            ranking = root / "expert-M.bin"
            ranking.write_bytes(b"seed")
            pins = dict(
                models=dict(files=[dict(rfilename=model.name, size=5)]),
                expert_profiles=dict(
                    M=dict(size=4, sha256=hashlib.sha256(b"seed").hexdigest())
                ),
            )
            (root / "pins.json").write_text(json.dumps(pins))
            cfg = dict(
                exe=str(root / "build-engine/strata.exe"),
                cwd=str(source),
                tokenizer=str(pack / "tokenizer"),
                args=[
                    "--native",
                    str(model),
                    "--pack",
                    str(pack),
                    "--expert-profile",
                    str(ranking),
                    "--max-context",
                    "262144",
                ],
                _recipe_weight_mtime_ns=model.stat().st_mtime_ns,
                _recipe_pack_files={
                    "index.txt": dict(
                        size=7, mtime_ns=(pack / "index.txt").stat().st_mtime_ns
                    )
                },
            )
            with patch.object(runtime, "ROOT", root):
                runtime.check_config(cfg)
                for key, value in [
                    ("exe", str(root / "other.exe")),
                    ("cwd", str(root / "other-source")),
                    ("tokenizer", str(root / "other-tokenizer")),
                    ("_recipe_weight_mtime_ns", 0),
                ]:
                    bad = dict(cfg, **{key: value})
                    with self.assertRaises(RuntimeError):
                        runtime.check_config(bad)
                duplicate = copy.deepcopy(cfg)
                duplicate["args"] += ["--native", str(model)]
                with self.assertRaises(ValueError):
                    runtime.check_config(duplicate)
                (pack / "index.txt").write_bytes(b"changed file")
                with self.assertRaisesRegex(RuntimeError, "Pack changed"):
                    runtime.check_config(cfg)
                (pack / "index.txt").write_bytes(b"fixture")
                cfg.pop("_recipe_pack_files")
                ranking.write_bytes(b"bad!")
                with self.assertRaisesRegex(RuntimeError, "Expert profile changed"):
                    runtime.check_config(cfg)


if __name__ == "__main__":
    unittest.main()
