@echo off
echo Starting Production Welcome System...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if required packages are installed
python -c "import ultralytics, cv2, pygame, psutil, mutagen" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Start the system
echo Starting welcome system...
python production_welcome_system.py

pause
