@echo off
title HealthAI Platform
echo ===================================================
echo           Starting HealthAI Platform
echo ===================================================
echo.
cd /d "C:\Users\arisha khan\.gemini\antigravity\scratch\health_ai_app"

echo Opening browser at http://localhost:8000 ...
start "" "http://localhost:8000"

echo Running server on http://localhost:8000 ...
"C:\Users\arisha khan\.conda\envs\yolo\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

pause
