@echo off
REM Run the document analysis system with sample input

REM Set environment
set SCRIPT_DIR=%~dp0
cd %SCRIPT_DIR%

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed
    exit /b 1
)

REM Create a virtual environment if it doesn't exist
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if they're not already installed
echo Installing requirements...
pip install -r requirements.txt

REM Run the document analysis system
echo Running document analysis system...
python -m src.main --input data\input.json --output output\result.json

REM Check if the analysis was successful
if %ERRORLEVEL% EQU 0 (
    echo Analysis completed successfully!
    echo Output saved to output\result.json
) else (
    echo Error: Analysis failed
    exit /b 1
)

REM Optionally run the performance profiler
if "%1"=="--profile" (
    echo Running performance profiler...
    python -m src.profiler --full-profile
)

REM Deactivate virtual environment
deactivate

echo Done! 