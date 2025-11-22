@echo off
echo ========================================
echo  Person Detection System Setup
echo ========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo [4/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup Complete! 
echo ========================================
echo.
echo To run the system:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run the system: python production_welcome_system.py
echo.
echo To start the system directly, use: start_system.bat
echo.
pause
