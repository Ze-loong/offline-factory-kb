@echo off
rem One-click start: run app.py (opens http://127.0.0.1:7860 in browser)
rem Python lookup order: .venv in this folder -> shared .venv three levels up -> python on PATH
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\..\.venv\Scripts\python.exe" (
    set "PY=%~dp0..\..\..\.venv\Scripts\python.exe"
)
"%PY%" app.py
pause
