@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0Download-Gyro-M.ps1" %*
if errorlevel 1 (
  echo.
  echo The model download did not finish. Read the error above; interrupted downloads can resume.
  if not defined GYRO_NO_PAUSE pause
  exit /b 1
)
echo.
echo Next, double-click INSTALL.cmd in this folder.
if not defined GYRO_NO_PAUSE pause
exit /b 0
