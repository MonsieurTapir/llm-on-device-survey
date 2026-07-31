@echo off
rem Double-clickable wrapper: bootstraps the latest release kit and runs it.
rem Works saved on its own, with no clone - it fetches run.ps1 if it isn't here.
rem Flat on purpose: raw.githubusercontent.com serves this file with LF endings,
rem and cmd.exe parses LF-only batch reliably only without blocks or labels.
setlocal
set "PS1=%~dp0run.ps1"
set "RAW=https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1"

if not exist "%PS1%" echo -^> downloading run.ps1
if not exist "%PS1%" powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '%RAW%' -OutFile '%PS1%'"
if not exist "%PS1%" echo.
if not exist "%PS1%" echo XX could not download run.ps1 - check your network connection and re-run.

rem Unblock-File first: a .ps1 downloaded from the internet carries
rem mark-of-the-web, and Windows prompts for confirmation (or refuses outright)
rem whenever a machine policy overrides the -ExecutionPolicy switch here.
if exist "%PS1%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%PS1%' -ErrorAction SilentlyContinue; & '%PS1%'"
pause
