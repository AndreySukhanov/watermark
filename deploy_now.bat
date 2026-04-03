@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo Обновление GPU сервера через git...
echo ==============================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_now.ps1"

if %errorlevel% neq 0 (
    echo [Ошибка] Не удалось обновить сервер через git.
    pause
    exit /b %errorlevel%
)

echo.
pause
