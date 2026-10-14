@echo off
setlocal DisableDelayedExpansion
set "GYRO_RECIPE_ROOT=%~dp0"
if defined GYRO_PYTHON goto ready
where py >nul 2>&1
if errorlevel 1 goto fallback
for %%V in (3.14 3.13 3.12) do (
  if not defined GYRO_PYTHON for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "GYRO_PYTHON=%%P"
)
if defined GYRO_PYTHON goto ready
:fallback
set "GYRO_PYTHON=python"
:ready
"%GYRO_PYTHON%" "%GYRO_RECIPE_ROOT%setup.py" %*
set "GYRO_EXIT=%errorlevel%"
if not "%GYRO_EXIT%"=="0" if not defined GYRO_NO_PAUSE pause
exit /b %GYRO_EXIT%
