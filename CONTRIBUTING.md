# Contributing

Thanks for helping make this project more useful and safer to evolve. The goal is to keep experiments fast while keeping the main branch reproducible.

## Working Model

Use short-lived branches and pull requests. Do not commit directly to `main`.

The active project workflow is organized into three workstreams:

- Data and evaluation foundation.
- Model and research validation.
- Decision product and collaboration layer.

See [`docs/workflow.md`](docs/workflow.md) before starting new work.

Recommended branch names:

- `feature/<short-name>` for user-facing features.
- `fix/<short-name>` for bug fixes.
- `chore/<short-name>` for repo, docs, CI or cleanup.
- `experiment/<short-name>` for research work that may not ship.

## Local Setup

On Windows:

```powershell
.\scripts\setup_environment.ps1 -WithData -WithML
```

Run the full local check before opening a PR:

```powershell
.\scripts\run_checks.ps1
```

Or manually:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Pull Request Rules

Every PR should include:

- A clear summary of what changed.
- The exact command used to test the change.
- Any generated metrics if model behavior changed.
- Any data assumptions or limitations.
- Confirmation that generated data, checkpoints and reports were not committed unless intentionally discussed.

## Data And Artifacts

Do not commit new generated files from:

- `Data/raw/`
- `Data/processed/`
- `artifacts/`
- `reports/`

If a dataset is needed for collaboration, document where it came from and how to regenerate it. Prefer scripts and reproducible commands over binary blobs.

## Financial Safety

Outputs are research signals, not financial advice. Any PR that changes signals, thresholds, costs, model training or dashboard recommendations should state the expected behavioral impact and the residual risk.
