@echo off
rem ---------------------------------------------------------------------------
rem  Run any Python script with the project's own virtual environment.
rem  No activation needed, and it is not affected by PowerShell execution policy.
rem
rem  Examples:
rem     run.cmd train.py --algo ppo --seed 0 --steps 250000
rem     run.cmd evaluate.py --run runs\ppo_s0 --episodes 10
rem     run.cmd aggregate.py
rem     run.cmd -m unittest discover -s tests
rem ---------------------------------------------------------------------------
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [run.cmd] virtual environment not found: %PY%
  echo [run.cmd] create it first:  powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
  exit /b 1
)
"%PY%" %*
exit /b %ERRORLEVEL%
