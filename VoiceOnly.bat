@echo off
rem VoiceOnly app: Normal / Voice only switch (Ctrl+Alt+V from any app).
rem Short venv path: avoids the Windows 260-character path limit for CUDA DLLs.
set UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Temp\vo-venv
cd /d "%~dp0"
rem --inexact keeps optional model packages (roformer, metrics) installed.
uv sync --inexact --extra demucs -q || echo Update skipped (VoiceOnly may already be running).
start "" "%UV_PROJECT_ENVIRONMENT%\Scripts\voiceonly-app.exe"
