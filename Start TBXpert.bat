@echo off
title TBXpert - TB screening
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment...
    python -m venv .venv || goto :error
)

".venv\Scripts\python.exe" -c "import flask, sklearn, pandas, h5py, PIL, waitress" 2>nul
if errorlevel 1 (
    echo Installing the required packages ^(first run only, needs internet^)...
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt waitress || goto :error
)

".venv\Scripts\python.exe" run_local.py
goto :end

:error
echo.
echo Something went wrong. Take a screenshot of this window.

:end
pause
