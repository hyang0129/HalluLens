@echo off
REM Remote GPU Connection Script for HalluLens Development (Windows)
REM This script automatically finds and connects to the GPU node running job "nb8887new"

echo.
echo 🚀 HalluLens Remote GPU Connection Script
echo ===========================================
echo.

REM Check if Git Bash is available
where bash >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Git Bash not found in PATH
    echo 💡 Please install Git for Windows or run this from Git Bash
    echo    Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Run the bash script
echo 🔄 Launching connection script via Git Bash...
echo.
bash connect_gpu.sh

echo.
echo ✅ Script execution completed.
pause
