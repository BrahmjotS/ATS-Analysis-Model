@echo off
echo Starting ATS Resume Analyzer...
echo.

REM Check if virtual environment exists
if not exist "venv311\Scripts\python.exe" (
    echo Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment and run app
call venv311\Scripts\activate.bat
python app.py
pause

