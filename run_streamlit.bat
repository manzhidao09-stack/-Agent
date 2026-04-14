@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Streamlit...
echo If this fails, run: python -m pip install -r requirements.txt
python -m streamlit run app.py
if errorlevel 1 pause
