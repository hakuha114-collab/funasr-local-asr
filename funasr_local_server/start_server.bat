@echo off
REM Launch local FunASR WebSocket service (no Docker, uses conda env "funasr")
REM Models load from the project-local cache; no download on first run.
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
REM Keep this folder to make the setup portable to another machine.
set "MODELSCOPE_CACHE=%~dp0..\models\modelscope"

REM Use UTF-8 in this console so Chinese transcription logs are not garbled.
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

echo MODELSCOPE_CACHE=%MODELSCOPE_CACHE%
python funasr_wss_server.py --port 10095 --device cpu --ngpu 0 --certfile ""
REM Optional: use local ONNX models in repo to skip first-time download.
REM Remove REM below and comment out the line above to enable:
REM python funasr_wss_server.py --port 10095 --device cpu --ngpu 0 --certfile "" --asr_model ..\models\damo\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx
pause
endlocal
