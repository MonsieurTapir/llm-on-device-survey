# One-command benchmark (Windows x64): pull the latest release kit from GitHub
# (checksum-verified), unpack it next to this script, and hand off to the
# kit's own run.ps1. Double-click run.bat, or standalone without a clone:
#   irm https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1 | iex
# Safe to re-run: an already-unpacked kit is reused, and the kit itself resumes.
$ErrorActionPreference = "Stop"
# Windows PowerShell renders a progress bar per response chunk, which costs more
# than the transfer itself on a kit-sized download.
$ProgressPreference = "SilentlyContinue"
$repo = "MonsieurTapir/llm-on-device-survey"
function Fail($msg) { Write-Host "`n!! $msg" -ForegroundColor Red; exit 1 }

if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
  Fail "no prebuilt kit for this architecture ($env:PROCESSOR_ARCHITECTURE) - Windows x64 only"
}

$tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest").tag_name
if (-not $tag) { Fail "cannot resolve the latest release - check your network connection" }

# Must match packaging/package.sh's STAGE_NAME — the release URL is built from it.
$kit = "llm-on-device-survey-$($tag -replace '^v', '')-windows-x64"
if (-not (Test-Path $kit)) {
  $base = "https://github.com/$repo/releases/download/$tag"
  Write-Host "== downloading $kit.zip"
  Invoke-WebRequest -Uri "$base/$kit.zip" -OutFile "$kit.zip" -UseBasicParsing
  # To a file, then read as text: GitHub serves every release asset as
  # application/octet-stream, and for a non-text content type Invoke-WebRequest
  # hands back .Content as a byte[] - which -split would turn into byte values.
  Invoke-WebRequest -Uri "$base/$kit.zip.sha256" -OutFile "$kit.zip.sha256" -UseBasicParsing
  $expected = ((Get-Content "$kit.zip.sha256" -Raw) -split '\s+')[0]
  $actual = (Get-FileHash "$kit.zip" -Algorithm SHA256).Hash
  # -ne on strings is case-insensitive: sha256sum writes lowercase, Get-FileHash upper.
  if ($actual -ne $expected) { Fail "checksum mismatch (partial download?) - delete $kit.zip and re-run" }
  Expand-Archive -Path "$kit.zip" -DestinationPath .
  # Clear mark-of-the-web the zip download stamped on the extracted files.
  Get-ChildItem -Path $kit -Recurse -File | Unblock-File
  Remove-Item "$kit.zip", "$kit.zip.sha256"
}

if ($env:BENCH_BOOTSTRAP_ONLY) { Write-Host "== bootstrap ok: $kit"; exit 0 }
& (Join-Path $kit "run.ps1")
