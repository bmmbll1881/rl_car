# Run any Python script with the project's own virtual environment (same as run.cmd).
#
#   .\run.ps1 train.py --algo ppo --seed 0 --steps 250000
#   .\run.ps1 aggregate.py
#
# If PowerShell refuses to run scripts ("execution policy"), use one of:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 train.py ...
#   or simply use run.cmd (batch files are not affected by that policy)
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "virtual environment not found: $py`ncreate it first: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1"
    exit 1
}
& $py @Args
exit $LASTEXITCODE
