@echo off
rem VoiceOnly app: Normal / Voice only switch (Ctrl+Alt+V from any app).
rem Short venv path: avoids the Windows 260-character path limit for CUDA DLLs.
set UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Temp\vo-venv
cd /d "%~dp0"
uv sync --extra demucs -q || (echo Setup failed & pause & exit /b 1)
start "" "%UV_PROJECT_ENVIRONMENT%\Scripts\voiceonly-app.exe"
