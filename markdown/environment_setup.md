# Environment setup

This project supports two equivalent dependency entry points:

- `pyproject.toml` for editable installs and package metadata.
- `requirements*.txt` for direct pip installs and deployment-friendly environments.

## Recommended Windows setup

From the repository root:

```powershell
.\scripts\setup_environment.bat -WithML
```

This creates `.venv`, upgrades `pip`, installs the project in editable mode with development dependencies, installs CPU PyTorch, compiles Python files, and runs the test suite.

If you only need the API, baseline, HMM, and dataset utilities without PyTorch:

```powershell
.\scripts\setup_environment.bat
```

To skip tests during installation:

```powershell
.\scripts\setup_environment.bat -WithML -SkipTests
```

## Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

## Requirements files

- `requirements.txt`: runtime API/data/HMM dependencies.
- `requirements-dev.txt`: runtime plus testing dependencies.
- `requirements-ml.txt`: development plus PyTorch for FINN training.

The recommended route remains `setup_environment.bat` because it handles the Windows PowerShell execution policy and installs PyTorch from the CPU wheel index explicitly.
