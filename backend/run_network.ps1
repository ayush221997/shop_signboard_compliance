# FastAPI on all interfaces so phones on your LAN can reach it (port 8000).
# From repo root you can also run:
#   .\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Set-Location $PSScriptRoot
$py = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Error "Python venv not found at $py"
  exit 1
}
& $py -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
