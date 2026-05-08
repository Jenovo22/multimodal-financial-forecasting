# Data Policy

The project uses market, macro and option-chain data. These files can grow quickly and may change often, so collaboration should be based on reproducible commands rather than committing new generated data.

## Versioned Data

Some historical data files are already tracked in Git. Treat them as legacy seed/reference data until the team decides to migrate them to external storage.

Do not remove or rewrite tracked data in the same PR as model/code changes unless the PR is explicitly a data migration.

## Generated Data

New generated data should stay local and is ignored by `.gitignore`:

- `Data/raw/`
- `Data/processed/`
- `artifacts/`
- `reports/`

If collaborators need the same generated file, document:

- Source command.
- Source date/time.
- Symbol/universe.
- Filters.
- Expected row count.
- Checksum if the file is large or shared externally.

For option snapshots, generate a combined manifest with:

```powershell
.\.venv\Scripts\python.exe scripts\combine_option_snapshots.py `
  --source-dir Data\raw\options `
  --output Data\processed\combined_option_snapshots.csv `
  --manifest-output Data\processed\combined_option_snapshots_manifest.json
```

The manifest should be used in PR descriptions instead of committing generated datasets.

## Model Artifacts

Checkpoints such as `artifacts/finn_model.pt` are local outputs. Do not commit them unless the team explicitly creates a release process for model artifacts.

Recommended pattern:

```text
artifacts/
  finn_model.pt
  finn_model_selection.pt
reports/
  finn_training_metrics.json
  prediction_dashboard.html
```

## Secrets

Do not commit `.env`, API keys, broker credentials or paid-data credentials. Use `.env.example` for non-secret defaults.

## Future Recommendation

If data size grows, introduce one of these:

- Cloud object storage for datasets and checkpoints.
- DVC for reproducible dataset versioning.
- GitHub Releases for frozen small artifacts.
