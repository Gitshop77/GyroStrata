"""HIP selection, adapter telemetry and resource boundaries without an AMD GPU."""
import json
import pathlib
import argparse
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import amd
import amd_setup
from hardware import adapt


def device(**changes):
    data = dict(index=1, name='Test Radeon', arch='gfx1101', luid='00000000_0000abcd',
                integrated=False, wave_size=32, total_mib=12288)
    data.update(changes)
    return data


class HipPolicy(unittest.TestCase):
    def test_runtime_index_and_luid_are_preserved(self):
        gpu = amd.parse_devices(json.dumps([device(index=0, integrated=True), device()]), 1)
        self.assertEqual(gpu['index'], 1)
        self.assertEqual(gpu['backend'], 'hip')
        self.assertEqual(gpu['uuid'], 'HIP-00000000_0000abcd')

    def test_invalid_hardware_fails_closed(self):
        for change in [dict(arch='gfx1103'), dict(integrated=True), dict(wave_size=64),
                       dict(total_mib=8192), dict(luid='00000000_00000000'), dict(luid='bad')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                amd.parse_devices(json.dumps([device(**change)]), 1)
        for payload in ['{}', '[]', json.dumps([device(), device()])]:
            with self.assertRaises(ValueError):
                amd.parse_devices(payload, 1)

    def test_telemetry_counts_only_the_selected_adapter(self):
        samples = [('luid_0x00000000_0x0000ABCD_phys_0', 3 * 2**30),
                   ('luid_0x00000000_0x0000abcd_phys_1', 2**20),
                   ('luid_0x00000000_0x0000ffff_phys_0', 10 * 2**30)]
        self.assertEqual(amd.memory_used('00000000_0000abcd', samples), 3073)
        for samples in [[], [('luid_0x00000000_0x0000abcd_phys_0', -1)]]:
            with self.assertRaises(RuntimeError):
                amd.memory_used('00000000_0000abcd', samples)

    def test_hip_does_not_require_cuda_or_nvidia_driver(self):
        gpu = amd.parse_devices(json.dumps([device()]), 1)
        gpu.update(used_mib=500, driver='AMD display driver')
        hardware = dict(gpu=gpu, physical_cores=8, avx2=True, ram_gib=31.9, available_ram_gib=26)
        profile = json.loads((ROOT / 'profiles/M.json').read_text())
        cfg, env, settings = adapt(profile, {'CUDA_VISIBLE_DEVICES':'GPU-unrelated'}, hardware)
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '1')
        self.assertNotIn('--pcie-frac', cfg['args'])
        self.assertEqual(settings['pcie_fraction'], 'auto')
        self.assertEqual(settings['ram_experts_gib'], 13)
        self.assertEqual(settings['vram_reserve_mib'], 2048)
        self.assertLessEqual(settings['gpu_limit_mib'], 11776)
        self.assertEqual(cfg['args'][cfg['args'].index('--max-context')+1], '262144')
        cfg, _, settings = adapt(profile, {}, hardware, tuning={'pcie_fraction':0.4})
        self.assertEqual(settings['pcie_fraction'], 0.4)
        self.assertEqual(cfg['args'][cfg['args'].index('--pcie-frac')+1], '0.4')

    def test_runtime_mask_and_sdk_dll_path(self):
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'x','HIP_VISIBLE_DEVICES':'7','ROCR_VISIBLE_DEVICES':'8'}):
            env = amd.runtime_environment(ROOT, unmask=True)
        for name in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:
            self.assertNotIn(name, env)
        self.assertEqual(env['HIP_PLATFORM'], 'amd')
        self.assertTrue(env['PATH'].startswith(str(ROOT / 'bin')))

    def test_cmake_paths_keep_spaces_and_shell_characters(self):
        sdk = pathlib.Path('C:/SDK Folder % test !')
        text = amd_setup.cmake_cache(sdk, sdk / 'bin/clang++.exe', pathlib.Path('C:/Engine Folder/sources/llama'))
        self.assertIn('--rocm-path="C:/SDK Folder % test !"', text)
        self.assertIn('[=[C:/Engine Folder/sources/llama]=]', text)

    def _install(self, parity_failure=False):
        with tempfile.TemporaryDirectory(prefix='HIP spaces % ! ') as folder:
            root = pathlib.Path(folder)
            sdk = root / 'ROCm SDK'
            (sdk / 'lib/llvm/amdgcn/bitcode').mkdir(parents=True)
            (sdk / 'bin').mkdir()
            for name in ['amdhip64_7.dll','amd_comgr.dll']:
                (sdk / 'bin' / name).write_bytes(b'runtime')
            (root / 'profiles').mkdir()
            (root / 'profiles/M.json').write_bytes((ROOT / 'profiles/M.json').read_bytes())
            modeldir = root / 'Model Folder'
            modeldir.mkdir()
            model = modeldir / 'weights.gguf'
            model.write_bytes(b'weights')
            (root / 'pins.json').write_text(json.dumps(dict(strata=dict(revision='s',patch_sha256='p'),
                llama=dict(revision='l'),models=dict(files=[dict(rfilename=model.name,size=7,lfs=dict(sha256='sha'))]))))
            calls=[];phases=[]
            def run(command, **kwargs):
                calls.append(command)
                if parity_failure and pathlib.Path(command[0]).name=='native_expert_parity.exe':
                    raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0)
            def batch(script,env,**kwargs):
                phases.append(dict(env))
                target=root/('build-amd-probe' if env.get('GYRO_HIP_PROBE') else 'build-engine-hip')
                target.mkdir(exist_ok=True)
                (target/('gyro-amd-probe.exe' if env.get('GYRO_HIP_PROBE') else 'strata.exe')).write_bytes(b'built')
            hardware=dict(gpu=device(used_mib=100,backend='hip',uuid='HIP-test'),
                          physical_cores=8,avx2=True,ram_gib=31.9,available_ram_gib=26)
            options=argparse.Namespace(hip_root=str(sdk),check=False,model_dir=str(modeldir),gpu=1,
                workers=None,skip_weight_hash=False,api_key=False,profile_dir=None,pack_m=None)
            with mock.patch.object(amd_setup,'ROOT',root), mock.patch.object(amd_setup,'sdk_root',return_value=(sdk,sdk/'bin/clang++.exe')), \
                 mock.patch('setup.vcvars',return_value='vcvars64.bat'), mock.patch('setup.select_model_directory',return_value=modeldir), \
                 mock.patch('runtime.require_free_port'), mock.patch('runtime.run_batch',side_effect=batch), \
                 mock.patch('recipe.sha',return_value='sha'), mock.patch.object(amd_setup,'memory_used',return_value=100), \
                 mock.patch('hardware.resource_info',return_value=hardware), mock.patch.object(amd_setup.subprocess,'run',side_effect=run), \
                 mock.patch.object(amd_setup.subprocess,'check_output',return_value=json.dumps([device()])):
                if parity_failure:
                    with self.assertRaises(subprocess.CalledProcessError):
                        amd_setup.install(options)
                    self.assertFalse((root/'build-info-hip.json').exists())
                    self.assertFalse(any(len(command)>2 and pathlib.Path(command[1]).name=='recipe.py' and command[2]=='install' for command in calls))
                else:
                    amd_setup.install(options)
                    build=json.loads((root/'build-info-hip.json').read_text())
                    self.assertEqual(build['gpu_arch'],'gfx1101')
                    self.assertEqual(build['backend'],'hip')
                    self.assertIn('build-engine-hip/gyro-amd-probe.exe',build['binaries'])
                    self.assertIn('--backend',calls[-1])
                    self.assertIn('hip',calls[-1])
                    self.assertIn('--no-api-key',calls[-1])
                    self.assertEqual(phases[1]['GYRO_HIP_ARCH'],'gfx1101')
                    self.assertEqual(phases[0]['GYRO_HIP_PROBE'],'1')
                    self.assertFalse((root/'build-info.json').exists())

    def test_hip_install_wires_build_probe_parity_and_profile(self):
        self._install()

    def test_failed_quant_parity_creates_no_profile_or_build_provenance(self):
        self._install(parity_failure=True)


if __name__ == '__main__':
    unittest.main()
