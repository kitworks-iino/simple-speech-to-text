@echo off
setlocal
cd /d "%~dp0"
"%~dp0SimpleSpeechToText.exe" --no-pause
set "STT_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %STT_EXIT%
