@echo off
setlocal enabledelayedexpansion
REM Setup script for Windows
REM Usage:
REM   setup.bat          (CUDA)
REM   setup.bat --cpu    (CPU-only PyTorch)
REM   setup.bat --dev    (installs pytest/dev deps)

set MODE=cuda
set DEV=0

:parse
if "%~1"=="" goto done_parse
if /I "%~1"=="--cpu" set MODE=cpu
if /I "%~1"=="--dev" set DEV=1
shift
goto parse
:done_parse

echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip

if "%DEV%"=="1" (
  pip install -r requirements-dev.txt
) else (
  pip install -r requirements.txt
)

if /I "%MODE%"=="cpu" (
  echo Installing PyTorch CPU...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
  echo Installing PyTorch CUDA (cu121)...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo Verifying CUDA...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo Setup complete!
echo Activate: .venv\Scripts\activate.bat
echo Train:    python train.py --model polynomial
echo Test:     pytest
endlocal
