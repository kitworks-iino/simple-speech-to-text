param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
if (-not (Test-Path $VsWhere)) { throw "Visual Studio C++ build tools are required for portable packaging." }
$VisualStudio = & $VsWhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VisualStudio) { throw "No Visual Studio installation with C++ redistributable files was found." }
$Redist = Join-Path $VisualStudio "VC/Redist/MSVC"
$Candidates = Get-ChildItem $Redist -Directory -Recurse -Filter "Microsoft.VC*.CRT" |
    Where-Object { $_.Parent.Name -eq "x64" -and (Test-Path (Join-Path $_.FullName "msvcp140.dll")) } |
    Sort-Object { [version](Get-Item (Join-Path $_.FullName "msvcp140.dll")).VersionInfo.FileVersion.Split(" ")[0] } -Descending
$Crt = $Candidates | Select-Object -First 1
if (-not $Crt) { throw "The x64 Visual C++ app-local CRT was not found." }
$Internal = Join-Path $Destination "_internal"
$LicenseFolder = Join-Path $Destination "licenses/microsoft-runtime"
New-Item -ItemType Directory -Force -Path $Internal, $LicenseFolder | Out-Null
$Libraries = @(Get-ChildItem $Crt.FullName -File -Filter "*.dll")
$OpenMp = Get-ChildItem $Crt.Parent.FullName -Directory -Filter "Microsoft.VC*.OpenMP" | Select-Object -First 1
if ($OpenMp) { $Libraries += @(Get-ChildItem $OpenMp.FullName -File -Filter "*.dll") }
$Manifest = @()
foreach ($Library in $Libraries) {
    # Root serves the launcher; _internal serves Python's bundled DLL search path.
    Copy-Item $Library.FullName (Join-Path $Destination $Library.Name) -Force
    Copy-Item $Library.FullName (Join-Path $Internal $Library.Name) -Force
    $Manifest += @{
        name = $Library.Name
        version = $Library.VersionInfo.FileVersion
        sha256 = (Get-FileHash $Library.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        origin = "Visual Studio VC/Redist/MSVC/$($Crt.Parent.Parent.Name)/x64"
    }
}
foreach ($Required in @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
    if (-not (Test-Path (Join-Path $Internal $Required))) { throw "Missing required CRT: $Required" }
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $LicenseFolder "files.json") -Encoding utf8
$Terms = "https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/"
Invoke-WebRequest -Uri $Terms -OutFile (Join-Path $LicenseFolder "Microsoft-Visual-C-Runtime-License.html")
@"
Microsoft Visual C++ Runtime app-local deployment
License: $Terms
Redistribution: https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files
DLLs are copied only from Visual Studio's designated x64 redistribution directory.
The Windows 11 Universal CRT is an operating-system component and is not bundled.
See files.json for original runtime file versions and SHA-256 hashes.
"@ | Set-Content (Join-Path $LicenseFolder "README.txt") -Encoding utf8
Write-Host "Bundled $($Libraries.Count) Visual C++ runtime DLLs."
