@echo off
rem Double-clickable wrapper: bootstraps the latest release kit and runs it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
