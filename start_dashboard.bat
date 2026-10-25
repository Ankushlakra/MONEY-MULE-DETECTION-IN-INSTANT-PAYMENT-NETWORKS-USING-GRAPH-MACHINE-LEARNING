@echo off
cd /d "%~dp0"
echo Starting AML dashboard. Close this window or press Ctrl+C to stop it.
python -m streamlit run app.py
pause
