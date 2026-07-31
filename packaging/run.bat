@echo off
rem Double-clickable wrapper: runs the real entry point (run.ps1) and keeps the
rem window open so the final instructions stay readable.
rem Unblock-File first: unzipped by hand, run.ps1 carries mark-of-the-web, and
rem Windows prompts for confirmation (or refuses outright) whenever a machine
rem policy overrides the -ExecutionPolicy switch. run.ps1 clears the mark off
rem the rest of the kit itself.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%~dp0run.ps1' -ErrorAction SilentlyContinue; & '%~dp0run.ps1'"
pause
