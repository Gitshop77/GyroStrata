"""Resource boundaries across representative CPU/GPU/RAM combinations; no GPU required."""

import json, pathlib, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware import adapt


class ResourcePolicy(unittest.TestCase):
    def test_general_hosts(self):
        for arch, cores, available, total in [
            ("86", 6, 24, 12288),
            ("89", 20, 28, 12282),
            ("120", 16, 18, 12288),
            ("86", 8, 14, 12288),
            ("89", 12, 28, 16384),
        ]:
            h = dict(
                gpu=dict(
                    name="TEST",
                    total_mib=total,
                    used_mib=512,
                    arch=arch,
                    uuid="GPU-TEST",
                ),
                physical_cores=cores,
                ram_gib=31.9,
                available_ram_gib=available,
            )
            for name in ["M"]:
                original = json.loads((ROOT / "profiles" / f"{name}.json").read_text())
                before = json.dumps(original, sort_keys=True)
                cfg, env, choice = adapt(original, {}, h)
                self.assertLessEqual(choice["ram_experts_gib"], min(16, available - 10))
                self.assertLessEqual(choice["gpu_limit_mib"], 11776)
                self.assertEqual(
                    choice["vram_reserve_mib"], 1024 + max(0, total - 12288)
                )
                self.assertEqual(choice["pcie_fraction"], 0.75)
                self.assertEqual(
                    cfg["args"][cfg["args"].index("--pool-workers") + 1], "0"
                )
                self.assertEqual(
                    cfg["args"][
                        cfg["args"].index("--conversation-cache-min-free-mib") + 1
                    ],
                    "512",
                )
                self.assertGreaterEqual(choice["pool_tasks"], 64)
                self.assertLessEqual(choice["pool_tasks"], 192)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-TEST")
                self.assertEqual(json.dumps(original, sort_keys=True), before)

    def test_insufficient_resources_rejected(self):
        baseline = dict(
            gpu=dict(
                name="TEST", total_mib=12288, used_mib=512, arch="89", uuid="GPU-TEST"
            ),
            physical_cores=8,
            ram_gib=31.9,
            available_ram_gib=24,
        )
        profile = json.loads((ROOT / "profiles/M.json").read_text())
        for field, value in [("ram_gib", 16), ("available_ram_gib", 13)]:
            h = dict(baseline)
            h[field] = value
            with self.assertRaises(ValueError):
                adapt(profile, {}, h)
        for field, value in [("total_mib", 8192), ("arch", "61")]:
            h = dict(baseline, gpu=dict(baseline["gpu"]))
            h["gpu"][field] = value
            with self.assertRaises(ValueError):
                adapt(profile, {}, h)

    def test_persistent_tuning_and_memory_floor(self):
        h = dict(
            gpu=dict(
                name="TEST", total_mib=12288, used_mib=512, arch="89", uuid="GPU-TEST"
            ),
            physical_cores=6,
            ram_gib=31.9,
            available_ram_gib=18,
        )
        p = json.loads((ROOT / "profiles/M.json").read_text())
        cfg, env, choice = adapt(
            p,
            {},
            h,
            4,
            dict(
                pcie_fraction=0.5,
                pool_tasks=128,
                ram_experts_gib=16,
                vram_reserve_mib=1536,
            ),
        )
        self.assertEqual(choice["ram_experts_gib"], 8)
        self.assertEqual(choice["pcie_fraction"], 0.5)
        self.assertEqual(choice["pool_tasks"], 128)
        self.assertEqual(choice["vram_reserve_mib"], 1536)
        self.assertEqual(cfg["args"][cfg["args"].index("--pool-workers") + 1], "4")
        for tuning in [
            dict(pcie_fraction=float("nan")),
            dict(pcie_fraction=1.1),
            dict(pool_tasks=63),
            dict(ram_experts_gib=17),
            dict(vram_reserve_mib=512),
            dict(unknown=1),
        ]:
            with self.assertRaises(ValueError):
                adapt(p, {}, h, tuning=tuning)


if __name__ == "__main__":
    unittest.main()
