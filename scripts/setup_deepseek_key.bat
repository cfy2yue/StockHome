@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_deepseek_key.ps1" -SmokeTest
pause
