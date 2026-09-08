@echo off
setlocal enabledelayedexpansion
echo.
echo ============================================================
echo  Stopping FunASR services - ports 10095 10098 8099
echo ============================================================
call :free_port 10095
call :free_port 10098
call :free_port 8099
echo.
echo  Verifying ports are released:
call :check_port 10095
call :check_port 10098
call :check_port 8099
echo.
echo  Done.
pause
endlocal
goto :eof

REM Kill any process listening on the given TCP port (show what is killed)
:free_port
set "P=%~1"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%P% " ^| findstr "LISTENING"') do echo Killing port !P! PID=%%a & taskkill /F /PID %%a
goto :eof

REM Report whether the port is now free
:check_port
set "P=%~1"
set "FOUND="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%P% " ^| findstr "LISTENING"') do set "FOUND=%%a"
if defined FOUND (echo   port !P!: STILL OCCUPIED by PID !FOUND!) else echo   port !P!: FREE
goto :eof
