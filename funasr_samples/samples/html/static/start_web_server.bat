@echo off
REM Launch HTTP server for the ASR web UI (no Docker, uses conda env)
setlocal
set CONDA=%USERPROFILE%\miniforge3\Scripts\activate.bat
if not exist "%CONDA%" set CONDA=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA%" (
  echo Conda not found. Open a miniforge terminal and run this script manually.
  pause & exit /b 1
)
call "%CONDA%" funasr
cd /d "%~dp0"

python -m http.server 8099
pause
endlocal
