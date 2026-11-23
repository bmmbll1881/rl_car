# Create / rebuild the project's standalone virtual environment.
# It does NOT depend on DeepSeek Harness or Codex: the interpreter is the copy inside
# runtime\python, which ships with numpy + Pillow in its own site-packages.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Rebuild
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Python "C:\Python312\python.exe"
#
# Interpreter priority:
#   1. -Python <path>
#   2. runtime\python\python.exe  (CPython 3.12 bundled with the project)
#   3. system py -3 / python      (then dependencies come from pip)
param(
    [string]$Python = "",
    [switch]$Rebuild,
    [switch]$ForcePipInstall
)

$ErrorActionPreference = 'Stop'
$proj = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $proj '.venv'
$rt = Join-Path $proj 'runtime\python\python.exe'
$rtSite = Join-Path $proj 'runtime\python\Lib\site-packages'

function Find-BasePython {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "interpreter not found: $Explicit" }
        return $Explicit
    }
    if (Test-Path $rt) { return $rt }
    foreach ($cand in @('py', 'python', 'python3')) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            try {
                $ok = & $cmd.Source -c "import sys; print(sys.version_info >= (3, 8))" 2>$null
                if ($ok -match 'True') { return $cmd.Source }
            } catch { }
        }
    }
    throw "no Python 3.8+ found. Install Python or pass -Python <path>."
}

$base = Find-BasePython -Explicit $Python
Write-Host "[setup] base interpreter: $base"

if (Test-Path $venv) {
    if (-not $Rebuild) {
        $py = Join-Path $venv 'Scripts\python.exe'
        if (Test-Path $py) {
            Write-Host "[setup] virtual environment already exists: $venv"
            & $py -c "import numpy, PIL; from PIL import Image; print('[setup] numpy', numpy.__version__, '/ pillow', PIL.__version__)"
            Write-Host "[setup] pass -Rebuild to recreate it"
            exit 0
        }
        Write-Host "[setup] existing .venv is incomplete, recreating..."
    }
    Remove-Item -Recurse -Force $venv
}

# runtime\python already carries numpy+Pillow, so --system-site-packages makes the
# venv usable offline. A system Python gets its dependencies from pip instead.
$hasLocalDeps = (Test-Path (Join-Path $rtSite 'numpy')) -and (Test-Path (Join-Path $rtSite 'PIL'))
$venvArgs = @('-m', 'venv')
if ($hasLocalDeps -and -not $ForcePipInstall) { $venvArgs += '--system-site-packages' }
$venvArgs += $venv
& $base @venvArgs
if ($LASTEXITCODE -ne 0) { throw "failed to create the virtual environment" }

$vpy = Join-Path $venv 'Scripts\python.exe'
Write-Host "[setup] created: $venv"

& $vpy -c "import numpy, PIL; from PIL import Image" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] installing dependencies with pip (needs network)..."
    & $vpy -m pip install --upgrade pip
    & $vpy -m pip install -r (Join-Path $proj 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (offline?). Put the packages into runtime\python\Lib\site-packages and retry." }
} else {
    Write-Host "[setup] dependencies OK (offline, from runtime\python)"
}

& $vpy -c "import sys, numpy, PIL; from PIL import Image; print('[setup] OK  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pillow', PIL.__version__)"
Write-Host ""
Write-Host "next:"
Write-Host "  .\run.cmd train.py --algo ppo --seed 0 --steps 250000"
Write-Host "  or:  .\.venv\Scripts\python.exe train.py --algo ppo --seed 0 --steps 250000"
