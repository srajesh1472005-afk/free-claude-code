@echo off
echo Starting AI Smart Production Scheduler & Rescheduler...
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run src\free_claude_code\streamlit\app.py
pause
