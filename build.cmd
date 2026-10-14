@echo off
setlocal DisableDelayedExpansion
set "GYRO_RECIPE_ROOT=%~dp0"
cd /d "%GYRO_RECIPE_ROOT%"
if defined GYRO_VCVARS call "%%GYRO_VCVARS%%" >nul
if errorlevel 1 exit /b 1
where cl >nul 2>&1
if errorlevel 1 (
 echo Open an x64 Native Tools Command Prompt for VS 2022, then run build.cmd.
 exit /b 1
)
if not defined GYRO_PYTHON set "GYRO_PYTHON=python"
if not exist ".venv\Scripts\python.exe" "%GYRO_PYTHON%" -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r "sources\strata\requirements.txt"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
set "GYRO_CMAKE=%GYRO_RECIPE_ROOT%.venv\Scripts\cmake.exe"
set "GYRO_NINJA=%GYRO_RECIPE_ROOT%.venv\Scripts\ninja.exe"
set "GYRO_CUDA_ARCH="
for /f "delims=" %%A in ('.venv\Scripts\python.exe hardware.py arch') do set "GYRO_CUDA_ARCH=%%A"
if not defined GYRO_CUDA_ARCH exit /b 1
"%GYRO_CMAKE%" -S sources\strata -B build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=%GYRO_CUDA_ARCH% -DCMAKE_MAKE_PROGRAM="%GYRO_NINJA%" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="%GYRO_RECIPE_ROOT%sources\llama"
if errorlevel 1 exit /b 1
"%GYRO_CMAKE%" --build build-engine --target strata draft_policy_test native_expert_parity kv_stream_parity async_job_test conv_cache_test file_expert_source_test tq_cpu_parity native_hadamard_test -j 4
if errorlevel 1 exit /b 1
build-engine\draft_policy_test.exe
if errorlevel 1 exit /b 1
build-engine\kv_stream_parity.exe
if errorlevel 1 exit /b 1
build-engine\async_job_test.exe
if errorlevel 1 exit /b 1
build-engine\conv_cache_test.exe
if errorlevel 1 exit /b 1
build-engine\file_expert_source_test.exe
if errorlevel 1 exit /b 1
build-engine\file_expert_source_test.exe --ram-rebalance-gpu
if errorlevel 1 exit /b 1
build-engine\native_hadamard_test.exe
if errorlevel 1 exit /b 1
build-engine\tq_cpu_parity.exe
if errorlevel 1 exit /b 1
set "STRATA_FORCE_AVX2=1"
build-engine\tq_cpu_parity.exe
if errorlevel 1 exit /b 1
set "STRATA_FORCE_AVX2="
cl /nologo /std:c++17 /EHsc /O2 /I sources\strata\include tests\policy-reprobe-test.cpp sources\strata\src\spec\draft_policy.cpp /Fe:build-engine\policy-reprobe-test.exe /Fo:build-engine\
if errorlevel 1 exit /b 1
build-engine\policy-reprobe-test.exe
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" runtime.py
exit /b %errorlevel%
