param(
  [string]$AppName = "ClothingMatcher",
  [string]$ModelFile = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
$ModelSource = if ($ModelFile) { $ModelFile } else { "models" }

$Python = Get-Command py -ErrorAction SilentlyContinue
$PythonArgs = @()
if ($Python) {
  $PythonArgs += "-3"
} else {
  $Python = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $Python) {
  throw "Python was not found. Install Python 3.10+ and ensure py.exe or python.exe is on PATH."
}

$PyInstallerArgs = @(
  "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--onedir",
  "--name", $AppName,
  "--hidden-import", "tkinter",
  "--collect-all", "tkinterdnd2",
  "--hidden-import", "onnxruntime",
  "--exclude-module", "torch",
  "--exclude-module", "torchvision",
  "--exclude-module", "pandas",
  "--exclude-module", "matplotlib",
  "--exclude-module", "scipy",
  "--exclude-module", "pyarrow",
  "--collect-all", "rapidocr_onnxruntime"
)
if (Test-Path -LiteralPath $ModelSource) {
  $PyInstallerArgs += @("--add-data", "$ModelSource;models")
  Write-Host "Including model data: $ModelSource"
} else {
  Write-Warning "Model data not found: $ModelSource. Building the lightweight fallback version."
}
$PyInstallerArgs += "app.py"

& $Python.Source @PythonArgs @PyInstallerArgs
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Copy-Item -LiteralPath "$ProjectDir\README.txt" -Destination "$ProjectDir\dist\$AppName\README.txt" -Force
Write-Host "Build complete: $ProjectDir\dist\$AppName"
