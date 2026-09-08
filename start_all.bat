@echo off
REM ===========================================================================
REM  One-click launcher: local FunASR service + diarization service + ASR page
REM  Opens three independent windows:
REM    [FunASR Service]      ws://localhost:10095   (mic realtime, wav only)
REM    [Diarization Service] http://localhost:10098 (folder batch, all formats + spk)
REM    [ASR Web UI]          http://localhost:8099/asr_studio.html
REM  Main window opens browser; click "Connect" in the page.
REM ===========================================================================
setlocal
set ROOT=%~dp0

echo.
echo ============================================================
echo  Launching three windows...
echo    [FunASR Service]      ws://localhost:10095   (mic realtime)
echo    [Diarization Service] http://localhost:10098 (folder batch + spk)
echo    [ASR Web UI]          http://localhost:8099/asr_studio.html
echo.
echo  Models load from the local cache (models\modelscope). No download.
echo ============================================================

REM --- Free ports used by our services (kill stale processes) ---
call :free_port 10095
call :free_port 10098
call :free_port 8099

REM --- Window 1: ASR WebSocket service (mic realtime) ---
start "FunASR Service" "%ROOT%funasr_local_server\start_server.bat"

REM --- Window 2: diarization HTTP service (folder batch + speaker) ---
start "Diarization Service" "%ROOT%funasr_local_server\start_diarize_server.bat"

REM --- Window 3: web page hosting ---
start "ASR Web UI" "%ROOT%funasr_samples\samples\html\static\start_web_server.bat"

timeout /t 3 >nul
start "" "http://localhost:8099/asr_studio.html"
pause
endlocal
goto :eof

REM --- Kill any process listening on the given TCP port ---
:free_port
set "P=%~1"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%P% " ^| findstr "LISTENING"') do echo Freeing port %P% PID=%%a & taskkill /F /PID %%a >nul 2>&1
goto :eof
