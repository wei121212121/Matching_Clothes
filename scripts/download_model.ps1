param(
  [string]$Destination = "clothing_matcher_v8_ui_alt/models/sscd_disc_mixup.onnx"
)

$ErrorActionPreference = "Stop"
$ModelUrl = "https://github.com/wei121212121/Matching_Clothes/releases/download/model-v1/sscd_disc_mixup.onnx"
$ExpectedSha256 = "6bd5e2d6e5ecdc077d01dd2353806142c84c9aeea5d3304d581e741802923e4f"
$Target = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\$Destination"))
$Parent = Split-Path -Parent $Target

New-Item -ItemType Directory -Force -Path $Parent | Out-Null
Write-Host "Downloading SSCD ONNX model..."
Invoke-WebRequest -Uri $ModelUrl -OutFile $Target

$ActualSha256 = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
  Remove-Item -LiteralPath $Target -Force
  throw "Model checksum verification failed. The downloaded file was removed."
}

Write-Host "Model ready: $Target"
