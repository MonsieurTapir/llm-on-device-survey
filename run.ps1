# One-command benchmark (Windows x64): pull the latest release kit from GitHub
# (checksum-verified), unpack it into an llm-on-device-survey\ folder under
# the current directory (in place when the current directory already is one),
# and hand off to the kit's own run.ps1. Kits are versioned subfolders of
# llm-on-device-survey\ and share its models\ cache, so a new release skips
# the model download. Double-click run.bat, or standalone
# without a clone:
#   irm https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1 | iex
# Safe to re-run: an already-unpacked kit is reused, and the kit itself resumes.
$ErrorActionPreference = "Stop"
# Windows PowerShell renders a progress bar per response chunk, which costs more
# than the transfer itself on a kit-sized download.
$ProgressPreference = "SilentlyContinue"
$repo = "MonsieurTapir/llm-on-device-survey"
# Same visual language as the kit's run.ps1 and the survey CLI.
function Say($msg)  { Write-Host ""; Write-Host "-> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host "XX $msg" -ForegroundColor Red; exit 1 }

if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
  Fail "no prebuilt kit for this architecture ($env:PROCESSOR_ARCHITECTURE) - Windows x64 only"
}

$tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest").tag_name
if (-not $tag) { Fail "cannot resolve the latest release - check your network connection" }

# Must match packaging/package.sh's naming: the release URL is built from
# ARTIFACT, and the archive unpacks to llm-on-device-survey\<tag>-<target>\.
# When the current folder already IS llm-on-device-survey\ (a re-run from
# inside a previous download, or a repo checkout), unpack in place instead of
# nesting a second folder - and a second ~8 GB models cache - inside the first.
$asset = "llm-on-device-survey-$($tag -replace '^v', '')-windows-x64"
$root = "llm-on-device-survey"
if ((Split-Path -Leaf (Get-Location)) -eq $root) { $root = "." }
$kit = Join-Path $root "$tag-windows-x64"
if (-not (Test-Path $kit)) {
  $base = "https://github.com/$repo/releases/download/$tag"
  Say "downloading $asset.zip"
  Invoke-WebRequest -Uri "$base/$asset.zip" -OutFile "$asset.zip" -UseBasicParsing
  # To a file, then read as text: GitHub serves every release asset as
  # application/octet-stream, and for a non-text content type Invoke-WebRequest
  # hands back .Content as a byte[] - which -split would turn into byte values.
  Invoke-WebRequest -Uri "$base/$asset.zip.sha256" -OutFile "$asset.zip.sha256" -UseBasicParsing
  $expected = ((Get-Content "$asset.zip.sha256" -Raw) -split '\s+')[0]
  $actual = (Get-FileHash "$asset.zip" -Algorithm SHA256).Hash
  # -ne on strings is case-insensitive: sha256sum writes lowercase, Get-FileHash upper.
  if ($actual -ne $expected) { Fail "checksum mismatch (partial download?) - delete $asset.zip and re-run" }
  if ($root -eq ".") {
    # Expand-Archive cannot strip the archive's llm-on-device-survey\ prefix;
    # extract to a scratch dir and move the kit up.
    Expand-Archive -Path "$asset.zip" -DestinationPath ".kit-extract" -Force
    Move-Item (Join-Path ".kit-extract\llm-on-device-survey" "$tag-windows-x64") $kit
    Remove-Item ".kit-extract" -Recurse
  } else {
    # -Force: llm-on-device-survey\ already exists when an older kit shares it.
    Expand-Archive -Path "$asset.zip" -DestinationPath . -Force
  }
  # Clear mark-of-the-web the zip download stamped on the extracted files.
  Get-ChildItem -Path $kit -Recurse -File | Unblock-File
  Remove-Item "$asset.zip", "$asset.zip.sha256"
}

if ($env:BENCH_BOOTSTRAP_ONLY) { Write-Host "OK bootstrap ok: $kit" -ForegroundColor Green; exit 0 }
& (Join-Path $kit "run.ps1")
# `exit` inside a script invoked with & ends that script only — propagate its
# code, or a failed kit run reports success to whatever launched this.
exit $LASTEXITCODE
