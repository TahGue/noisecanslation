@echo off
rem VoiceOnly live: hear only voices from your PC (driver-free).
rem SINK   = output Windows plays to and you do NOT listen to (set it as Windows default output)
rem OUTPUT = where you listen (names from: uv run voiceonly devices)
set SINK=Realtek
set OUTPUT=WH-1000XM4
rem Short venv path: avoids the Windows 260-character path limit for CUDA DLLs.
set UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Temp\vo-venv
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%~dp0"
uv run voiceonly live -i "loopback:%SINK%" -o "%OUTPUT%" -m htdemucs
pause
