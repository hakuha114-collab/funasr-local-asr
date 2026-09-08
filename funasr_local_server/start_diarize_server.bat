@echo off
REM Launch local diarization HTTP service (cam++ speaker diarization, all formats)
REM Uses conda env "funasr". This is the reliable multi-speaker path the web
REM page's "folder batch" tab POSTs files to (port 10098).
setlocal
set CONDA=%USERPROFILE%\miniforge3\Scripts\activate.bat
if not exist "%CONDA%" set CONDA=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA%" (
  echo Conda not found. Open a miniforge terminal and run this script manually.
  pause & exit /b 1
)
call "%CONDA%" funasr
cd /d "%~dp0"

REM Point ModelScope cache to the project-local models\modelscope folder.
set "MODELSCOPE_CACHE=%~dp0..\models\modelscope"

REM Use UTF-8 in this console so Chinese transcription logs are not garbled.
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

echo MODELSCOPE_CACHE=%MODELSCOPE_CACHE%
REM Logs are shown directly in this window so you can see model loading
REM and "serving on port 10098". A copy is also saved to diarize_server.log.
python -u diarize_server.py --port 10098 --device cpu 2>&1 | tee diarize_server.log
pause
endlocal
