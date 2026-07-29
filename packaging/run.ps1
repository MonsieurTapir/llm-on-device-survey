# One-shot contributor entry point (Windows): check the exe runs here, set up a
# private Python, fetch models, measure, and pack a submission tarball. Safe to
# re-run — every step resumes or is idempotent. Nothing is installed outside
# this folder and uv's cache.
$ErrorActionPreference = "Stop"
# Every step below checks $LASTEXITCODE itself and explains what to do; on
# pwsh 7.4+ this preference would turn a nonzero native exit into a raw
# exception first, and the contributor would see a stack trace instead.
$PSNativeCommandUseErrorActionPreference = $false
Set-Location -Path $PSScriptRoot

function Say($msg)  { Write-Host "`n== $msg" }
function Fail($msg) { Write-Host "`n!! $msg" -ForegroundColor Red; exit 1 }

# Unzipped by hand, every file carries the mark Windows puts on downloads, and
# SmartScreen gates the exes on it. The bootstrap run.ps1 clears it after
# unpacking; a manual download has nobody to do it. Scoped to the executables
# (and this folder's own files) so re-runs don't walk the models or the venv.
Get-ChildItem -Path $PSScriptRoot -File | Unblock-File -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue `
  -Path (Join-Path $PSScriptRoot "bin"), (Join-Path $PSScriptRoot "backends") |
  Unblock-File -ErrorAction SilentlyContinue

$uv = Join-Path $PSScriptRoot "bin\uv.exe"
if (-not (Test-Path $uv)) { Fail "bin\uv.exe is missing - the zip may be incomplete; please re-download it." }
$exe = Join-Path $PSScriptRoot "backends\llamacpp\build\bench-llamacpp.exe"
if (-not (Test-Path $exe)) { Fail "the benchmark exe is missing - the zip may be incomplete; please re-download it." }

Say "Checking the benchmark exe starts on this machine"
# Start-Process, not `exe *> log`: the exe logs its device scan to stderr by
# contract, and PowerShell turns a *redirected* native command's stderr into
# NativeCommandError records - terminating under ErrorActionPreference Stop, so
# a healthy exe would abort the run. Here the OS writes the streams itself, and
# the log stays verbatim. Absolute paths: the redirect targets resolve against
# the process directory, which Set-Location does not move.
$versionLog = Join-Path $PSScriptRoot "exe-version.log"
$errorLog = Join-Path $PSScriptRoot "exe-error.log"
$check = Start-Process -FilePath $exe -ArgumentList "version" -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput $versionLog -RedirectStandardError $errorLog
if ($check.ExitCode -ne 0) {
  # An exit code and no stderr at all means the loader, not the exe: a missing
  # DLL never reaches our logging (0xc0000135 = STATUS_DLL_NOT_FOUND).
  Fail ("the benchmark exe failed to start (exit code $($check.ExitCode); details in " +
    "exe-error.log). Please open an issue and attach exe-error.log: " +
    "https://github.com/MonsieurTapir/llm-on-device-survey/issues/new")
}
Remove-Item $versionLog, $errorLog -ErrorAction SilentlyContinue

Say "Setting up Python (self-contained - nothing touches your system Python)"
& $uv sync --project harness
if ($LASTEXITCODE -ne 0) { Fail "Python setup failed - check your network connection and re-run run.bat (it resumes)." }

Say "Fetching models (about 8 GB on first run; safe to interrupt and re-run)"
& $uv run --project harness bench fetch
if ($LASTEXITCODE -ne 0) { Fail "model download failed or was interrupted - re-run run.bat to resume. Downloads can look stalled for minutes and then jump; that is normal." }

Say "Conformance-checking the exe against the contract"
& $uv run --project harness bench check --backend llamacpp
if ($LASTEXITCODE -ne 0) { Fail "the exe runs but failed its conformance check - please open an issue with the output above." }

Say "What will be measured on this machine"
& $uv run --project harness bench plan --backend llamacpp
if ($LASTEXITCODE -ne 0) { Fail "planning failed - please open an issue with the output above." }

Say "Running the benchmark (about 15 min on fast machines, up to an hour on slow ones; keep it plugged in and idle)"
& $uv run --project harness bench run --backend llamacpp --out results/local
if ($LASTEXITCODE -ne 0) { Fail "the benchmark run failed - please open an issue with the output above." }

Say "Packing your submission"
& $uv run --project harness bench bundle results/local --out .
if ($LASTEXITCODE -ne 0) { Fail "bundling failed - please open an issue with the output above." }

Say "All done - attach the submission-*.tar.gz above to a new issue (link above)."
