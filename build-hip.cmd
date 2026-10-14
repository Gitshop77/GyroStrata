@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not defined GYRO_HIP_ROOT exit /b 1
if not defined GYRO_HIP_CXX exit /b 1
if not defined GYRO_VCVARS exit /b 1
call "%GYRO_VCVARS%" >nul
if errorlevel 1 exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%GYRO_HIP_ROOT%"
set "ROCM_PATH=%GYRO_HIP_ROOT%"
set "PATH=%GYRO_HIP_ROOT%\bin;%GYRO_HIP_ROOT%\lib\llvm\bin;%CD%\.venv\Scripts;%PATH%"
if defined GYRO_HIP_PROBE goto probe
if not defined GYRO_HIP_ARCH exit /b 1
cmake -C .hip-config.cmake -S sources\strata -B build-engine-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%GYRO_HIP_ARCH%"
if errorlevel 1 exit /b 1
cmake --build build-engine-hip --target strata native_expert_parity kv_stream_parity file_expert_source_test tq_cpu_parity native_hadamard_test draft_policy_test async_job_test conv_cache_test -j 2
exit /b %errorlevel%
:probe
cmake -C .hip-config.cmake -S amd-probe -B build-amd-probe -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build-amd-probe --target gyro-amd-probe -j 2
exit /b %errorlevel%
