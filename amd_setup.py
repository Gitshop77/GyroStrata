"""Opt-in HIP build/install; retains the independent CUDA build and provenance."""
import hashlib
import json
import shutil
import subprocess
import sys

from amd import ROOT, parse_devices, runtime_environment, sdk_root, memory_used


def cmake_cache(sdk, compiler, llama):
    def quote(value):
        value = str(value).replace('\\', '/')
        level = '='
        while ']' + level + ']' in value:
            level += '='
        return '[' + level + '[' + value + ']' + level + ']'
    flags = f'--rocm-path="{sdk.as_posix()}" --rocm-device-lib-path="{(sdk / "lib/llvm/amdgcn/bitcode").as_posix()}"'
    entries = dict(CMAKE_CXX_COMPILER=compiler, CMAKE_C_COMPILER=compiler.with_name('clang.exe'),
                   CMAKE_HIP_COMPILER=compiler, CMAKE_HIP_COMPILER_ROCM_ROOT=sdk,
                   CMAKE_PREFIX_PATH=sdk, CMAKE_HIP_FLAGS=flags, STRATA_GGML_DIR=llama)
    return '\n'.join(f'set({key} {quote(value)} CACHE STRING "HIP install" FORCE)' for key, value in entries.items()) + '\n'


def install(options):
    from setup import select_model_directory, vcvars
    from runtime import require_free_port, run_batch
    from recipe import sha

    sdk, compiler = sdk_root(options.hip_root)
    if not (sdk / 'lib/llvm/amdgcn/bitcode').is_dir():
        raise RuntimeError('Use a complete ROCm development SDK, including AMDGPU bitcode.')
    compiler_env = vcvars()
    if options.check:
        print('HIP compiler and Visual Studio prerequisites found. Device/parity checks run after building.')
        return
    require_free_port('0.0.0.0', 8080)
    modeldir = select_model_directory(options.model_dir, ROOT / '.model-dir')
    pins = json.loads((ROOT / 'pins.json').read_text())
    weight = pins['models']['files'][0]
    model = modeldir / weight['rfilename']
    if not model.is_file() or model.stat().st_size != weight['size']:
        raise RuntimeError('Pinned Gyro-M GGUF is missing or incomplete; run DOWNLOAD-MODEL.cmd first.')
    stamp = model.stat().st_size, model.stat().st_mtime_ns
    if not options.skip_weight_hash and sha(model) != weight['lfs']['sha256']:
        raise RuntimeError('Wrong model SHA256; the existing weights were left untouched.')
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.is_file():
        subprocess.run([sys.executable, '-m', 'venv', str(ROOT / '.venv')], check=True)
    subprocess.run([str(python), str(ROOT / 'recipe.py'), 'sources'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'sources/strata/requirements.txt')], check=True)
    env = runtime_environment(sdk, unmask=True)
    env.update(GYRO_HIP_ROOT=str(sdk).replace('\\', '/'), GYRO_HIP_CXX=str(compiler).replace('\\', '/'),
               GYRO_VCVARS=compiler_env)
    (ROOT / '.hip-config.cmake').write_text(cmake_cache(sdk, compiler, ROOT / 'sources/llama'), encoding='utf-8')
    env['GYRO_HIP_PROBE'] = '1'
    run_batch(ROOT / 'build-hip.cmd', env=env, cwd=ROOT)
    env.pop('GYRO_HIP_PROBE')
    builddir = ROOT / 'build-engine-hip'
    builddir.mkdir(exist_ok=True)
    probe = builddir / 'gyro-amd-probe.exe'
    shutil.copy2(ROOT / 'build-amd-probe/gyro-amd-probe.exe', probe)
    # Avoid an incompatible runtime supplied by the display driver through System32.
    for name in ['amdhip64_7.dll', 'amd_comgr.dll']:
        path = sdk / 'bin' / name
        if not path.is_file():
            raise RuntimeError('The selected ROCm SDK is incomplete: ' + name)
        shutil.copy2(path, builddir / name)
    devices = subprocess.check_output([str(probe)], env=env, text=True, timeout=30)
    gpu = parse_devices(devices, options.gpu)
    gpu['used_mib'] = memory_used(gpu['luid'])
    from hardware import resource_info, adapt
    detected = resource_info(options.gpu, 'hip', detected_gpu=gpu)
    tuning = {name:getattr(options, name) for name in
              ['pcie_fraction','pool_tasks','ram_experts_gib','vram_reserve_mib']
              if getattr(options, name, None) is not None}
    adapt(json.loads((ROOT / 'profiles/M.json').read_text()), {}, detected, options.workers, tuning)
    env['GYRO_HIP_ARCH'] = gpu['arch']
    run_batch(ROOT / 'build-hip.cmd', env=env, cwd=ROOT)
    test_env = runtime_environment(sdk, {'HIP_VISIBLE_DEVICES':str(options.gpu)})
    for test, args in [('draft_policy_test', []), ('async_job_test', []), ('conv_cache_test', []),
                       ('file_expert_source_test', []), ('file_expert_source_test', ['--ram-rebalance-gpu']),
                       ('kv_stream_parity', []), ('native_hadamard_test', []), ('tq_cpu_parity', []),
                       ('native_expert_parity', [str(model)])]:
        subprocess.run([str(builddir / (test + '.exe')), *args], env=test_env, check=True)
    if (model.stat().st_size, model.stat().st_mtime_ns) != stamp:
        raise RuntimeError('Model changed during installation; rerun to verify it.')
    binary_paths = [*builddir.glob('*.exe'), *builddir.glob('*.dll')]
    build = dict(backend='hip', gpu_arch=gpu['arch'], hip_root=str(sdk),
                 engine='build-engine-hip/strata.exe',
                 strata_revision=pins['strata']['revision'], llama_revision=pins['llama']['revision'],
                 patch_sha256=pins['strata']['patch_sha256'],
                 binaries={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in binary_paths})
    (ROOT / 'build-info-hip.json').write_text(json.dumps(build, indent=2))
    (ROOT / '.model-dir').write_text(str(modeldir), encoding='utf-8')
    command = [str(python), str(ROOT / 'recipe.py'), 'install', '--backend', 'hip',
               '--model-dir', str(modeldir), '--gpu', str(options.gpu), '--skip-weight-hash']
    command.append('--api-key' if options.api_key else '--no-api-key')
    for name in ['profile_dir','pack_m','workers','pcie_fraction','pool_tasks','ram_experts_gib','vram_reserve_mib']:
        value = getattr(options, name, None)
        if value is not None:
            command += ['--' + name.replace('_','-'), str(value)]
    subprocess.run(command, check=True)
    print('Experimental HIP installation passed its local parity checks. Run the launcher beside the weights.')
