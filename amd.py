"""Experimental Windows HIP discovery and adapter-specific WDDM memory telemetry."""
import ctypes
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
ARCHS = {'gfx1030', 'gfx1031', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201'}
_counter = None


def sdk_root(value=None):
    value = value or os.environ.get('HIP_PATH') or os.environ.get('ROCM_PATH')
    if not value:
        raise RuntimeError('AMD needs a compatible ROCm development SDK with hipBLAS; pass --hip-root PATH.')
    root = pathlib.Path(value).resolve()
    compiler = root / 'lib/llvm/bin/clang++.exe'
    if not compiler.is_file():
        compiler = root / 'bin/clang++.exe'
    if not compiler.is_file() or not (root / 'include/hip/hip_runtime_api.h').is_file():
        raise RuntimeError('The selected HIP SDK lacks its compiler or development headers.')
    if not (root / 'bin/amdhip64_7.dll').is_file():
        raise RuntimeError('The selected HIP SDK lacks bin/amdhip64_7.dll. Select a complete ROCm SDK.')
    if not (root / 'bin/amd_comgr.dll').is_file() or not (root / 'include/hipblas/hipblas.h').is_file():
        raise RuntimeError('The selected ROCm SDK lacks amd_comgr or hipBLAS development files.')
    return root, compiler


def runtime_environment(root, values=None, unmask=False):
    env = os.environ.copy()
    for key, value in (values or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError('Environment keys and values must be JSON strings.')
        if value:
            env[key] = value
        else:
            env.pop(key, None)
    root = pathlib.Path(root)
    env.update(HIP_PLATFORM='amd', HIP_PATH=str(root), ROCM_PATH=str(root))
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('ROCR_VISIBLE_DEVICES', None)
    if unmask:
        env.pop('HIP_VISIBLE_DEVICES', None)
        env.pop('ROCR_VISIBLE_DEVICES', None)
    env['PATH'] = os.pathsep.join([str(root / 'bin'), str(root / 'lib/llvm/bin'), env.get('PATH', '')])
    return env


def parse_devices(text, index):
    devices = json.loads(text)
    if not isinstance(devices, list) or any(not isinstance(item, dict) for item in devices):
        raise ValueError('HIP probe returned an invalid device list.')
    selected = [item for item in devices if item.get('index') == index]
    if len(selected) != 1:
        raise ValueError('Selected HIP device was not found; rerun INSTALL.cmd with --gpu N.')
    gpu = dict(selected[0])
    if gpu.get('arch') not in ARCHS or gpu.get('integrated') or gpu.get('wave_size') != 32:
        raise ValueError('Select a supported discrete wave32 Radeon with --gpu N; an integrated device 0 may require --gpu 1.')
    if not re.fullmatch(r'[0-9a-f]{8}_[0-9a-f]{8}', str(gpu.get('luid', ''))) or gpu['luid'] == '00000000_00000000':
        raise ValueError('HIP did not provide a valid Windows adapter LUID; safe GPU monitoring is unavailable.')
    if type(gpu.get('total_mib')) is not int or gpu['total_mib'] < 11500:
        raise ValueError('This full-context profile requires at least 12 GiB dedicated VRAM.')
    gpu.update(backend='hip', uuid='HIP-' + gpu['luid'])
    return gpu


class AdapterCounters:
    """English PDH raw gauges; counts the selected adapter across all processes."""
    class Value(ctypes.Structure):
        class Number(ctypes.Union):
            _fields_ = [('large', ctypes.c_longlong), ('double', ctypes.c_double)]
        _fields_ = [('status', ctypes.c_ulong), ('number', Number)]

    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('AMD memory monitoring requires Windows WDDM counters.')
        self.pdh = ctypes.WinDLL('pdh')
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        self.pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._ok(self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        try:
            self._ok(self.pdh.PdhAddEnglishCounterW(self.query, r'\GPU Adapter Memory(*)\Dedicated Usage', 0, ctypes.byref(self.counter)))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _ok(code):
        if code != 0:
            raise RuntimeError(f'Windows GPU memory counter failed:0x{code & 0xffffffff:08x}.')

    def read(self):
        class Item(ctypes.Structure):
            _fields_ = [('name', ctypes.c_wchar_p), ('value', AdapterCounters.Value)]
        self._ok(self.pdh.PdhCollectQueryData(self.query))
        size, count = ctypes.c_ulong(), ctypes.c_ulong()
        code = self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x400, ctypes.byref(size), ctypes.byref(count), None)
        if code & 0xffffffff != 0x800007d2 or not 0 < size.value < 16 * 1024 * 1024:
            raise RuntimeError('Windows did not return usable GPU adapter memory counters.')
        data = ctypes.create_string_buffer(size.value)
        self._ok(self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x400, ctypes.byref(size), ctypes.byref(count), data))
        items = ctypes.cast(data, ctypes.POINTER(Item))
        return [(items[i].name, items[i].value.number.large) for i in range(count.value) if items[i].value.status in (0, 1)]

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = ctypes.c_void_p()


def memory_used(luid, samples=None):
    global _counter
    if samples is None:
        if _counter is None:
            _counter = AdapterCounters()
        samples = _counter.read()
    high, low = luid.split('_')
    pattern = re.compile(rf'^luid_0x{high}_0x{low}_phys_\d+$', re.I)
    values = [value for name, value in samples if pattern.fullmatch(name or '')]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError('The selected AMD adapter has no valid dedicated-memory telemetry; refusing an unguarded launch.')
    return (sum(values) + 2**20 - 1) // 2**20


def gpu_info(index=0):
    marker = ROOT / 'build-info-hip.json'
    if not marker.is_file():
        raise RuntimeError('HIP engine is not built; run INSTALL.cmd --backend hip --hip-root PATH.')
    build = json.loads(marker.read_text())
    probe = ROOT / 'build-engine-hip/gyro-amd-probe.exe'
    for name in ['gyro-amd-probe.exe', 'amdhip64_7.dll', 'amd_comgr.dll']:
        relative = 'build-engine-hip/' + name
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != build.get('binaries', {}).get(relative):
            raise RuntimeError('HIP probe/runtime differs from its build provenance; reinstall before probing.')
    text = subprocess.check_output([str(probe)], env=runtime_environment(build['hip_root'], unmask=True), text=True, timeout=30)
    gpu = parse_devices(text, int(index))
    gpu['used_mib'] = memory_used(gpu['luid'])
    return gpu
