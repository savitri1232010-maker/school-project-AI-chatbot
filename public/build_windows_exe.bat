@echo off
title Quickio AI — Windows .EXE Builder (CustomTkinter + YOLOv8 + Gemini/Ollama)
color 0D

echo ============================================================================
echo   QUICKIO AI - WINDOWS DESKTOP .EXE COMPILER (PyInstaller)
echo ============================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your Windows PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Installing required packages (CustomTkinter, google-genai, OpenCV, Ultralytics YOLO, PyInstaller)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [2/3] Compiling quickio_ai.py into standalone Windows executable (Quickio_AI.exe)...
python -m PyInstaller --noconfirm --onedir --windowed --name "Quickio_AI" --collect-all customtkinter --collect-all ultralytics quickio_ai.py

echo.
echo [3/3] Build Complete!
echo Your standalone Windows executable is ready at:
echo   dist\Quickio_AI\Quickio_AI.exe
echo.
pause
