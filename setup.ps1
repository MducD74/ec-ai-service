$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
  python -m venv $VenvPath
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

. $ActivateScript

python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Python virtual environment is ready at $VenvPath"
