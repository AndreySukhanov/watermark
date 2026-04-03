@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_now.ps1"
set "ERR=%ERRORLEVEL%"
exit /b %ERR%
