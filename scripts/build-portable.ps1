param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Repository = Split-Path $PSScriptRoot -Parent
Push-Location $Repository
try {
    if ($env:OS -ne "Windows_NT") { throw "Build on Windows x64." }
    & $Python -c "import struct,sys; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8, 'Standard Python 3.13 x64 is required'"
    if ($LASTEXITCODE -ne 0) { throw "Python version check failed." }
    & $Python scripts/audit_dependencies.py --environment --report build/dependency-environment.json
    if ($LASTEXITCODE -ne 0) { throw "Dependency audit failed." }
    & $Python -m PyInstaller --noconfirm --clean scripts/portable.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
    $Distribution = Join-Path $Repository "dist/SimpleSpeechToText"
    & $Python -m simple_speech_to_text --download-model --model-dir (Join-Path $Distribution "models/whisper-small")
    if ($LASTEXITCODE -ne 0) { throw "Verified model download failed." }
    # The public speech fixture is downloaded into a temporary directory and
    # is never copied into the distribution or the application's runtime data.
    & $Python scripts/smoke_speech.py --model-dir (Join-Path $Distribution "models/whisper-small") |
        Tee-Object -FilePath build/spoken-smoke.json
    if ($LASTEXITCODE -ne 0) { throw "Actual speech recognition and endpoint detection failed." }
    Copy-Item build/dependency-environment.json (Join-Path $Distribution "DEPENDENCY-ENVIRONMENT.json")
    & "$PSScriptRoot/bundle_vc_runtime.ps1" -Destination $Distribution
    & $Python scripts/collect_licenses.py --destination $Distribution
    if ($LASTEXITCODE -ne 0) { throw "License collection failed." }
    Copy-Item LICENSE (Join-Path $Distribution "LICENSE.txt")
    Copy-Item README.md (Join-Path $Distribution "README.md")
    New-Item -ItemType Directory -Force -Path (Join-Path $Distribution "docs") | Out-Null
    Copy-Item docs/dependencies.md (Join-Path $Distribution "docs/dependencies.md")
    Copy-Item scripts/Start.cmd (Join-Path $Distribution "Start.cmd")
    & $Python scripts/package_portable.py --directory $Distribution --output dist/simple-speech-to-text-windows-x64.zip
    if ($LASTEXITCODE -ne 0) { throw "Final archive audit failed." }
    & $Python scripts/smoke_portable.py --archive dist/simple-speech-to-text-windows-x64.zip
    if ($LASTEXITCODE -ne 0) { throw "Extracted executable model inference failed." }
} finally {
    Pop-Location
}
