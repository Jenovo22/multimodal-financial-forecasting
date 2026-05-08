# FINN Architecture Validation

This document defines the current validation protocol for the `experiment/model-architecture` branch.

## Goal

Validate FINN architecture changes without changing defaults blindly. Every architecture experiment must compare against Black-Scholes-Merton on the same train/validation/test split and must inspect basic Greek/no-arbitrage diagnostics.

## Current Recommended Baseline

The current baseline architecture is:

```text
prediction_mode = bsm_residual
hidden_dims = 64,64
activation = silu
residual_scale = 0.50
lambda_boundary = 0.1
lambda_pde = 0.0
lambda_arbitrage = 1.0
```

The `bsm_residual` mode starts from an analytical BSM anchor and learns a bounded residual:

```text
anchor = BSM(S, K, T, r, sigma_regime, q, option_type)
price = max(anchor + max(abs(anchor), residual_anchor_floor) * residual_scale * tanh(raw_output), 0)
```

This is safer than direct price learning while the dataset is small.

## Experiment Matrix

The architecture validation runner currently covers:

| Experiment | Hidden Dims | Activation | Residual Scale |
| --- | --- | --- | ---: |
| `baseline_64x64_silu_scale050` | `64,64` | `silu` | `0.50` |
| `compact_32x32_silu_scale050` | `32,32` | `silu` | `0.50` |
| `wide_128x128_silu_scale050` | `128,128` | `silu` | `0.50` |
| `baseline_64x64_gelu_scale050` | `64,64` | `gelu` | `0.50` |
| `conservative_64x64_silu_scale025` | `64,64` | `silu` | `0.25` |
| `flexible_64x64_silu_scale075` | `64,64` | `silu` | `0.75` |

## Commands

Run a quick smoke validation with two experiments:

```powershell
.\.venv\Scripts\python.exe scripts\run_finn_architecture_validation.py `
  --epochs 80 `
  --max-experiments 2
```

Run the full current matrix:

```powershell
.\.venv\Scripts\python.exe scripts\run_finn_architecture_validation.py `
  --epochs 300
```

Run only one named experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_finn_architecture_validation.py `
  --include baseline_64x64_silu_scale050 `
  --epochs 300
```

Outputs are written under:

```text
reports/finn_architecture_validation/
```

The runner writes:

- one `metrics.json` per experiment
- one `finn_split_predictions.csv` per experiment
- aggregate `summary.json`
- aggregate `summary.csv`

These generated outputs are local artifacts and should not be committed unless the team explicitly decides to publish a frozen experiment result.

## Metrics To Compare

Primary:

- `final_test_mae`
- `final_test_rmse`
- `baseline_test_mae`

Safety diagnostics:

- `delta_outside_bounds`
- `gamma_negative`
- validation best epoch
- train/validation/test gap

Promotion rule:

- Do not promote an architecture that only improves train error.
- Do not promote an architecture with worse test MAE than BSM.
- Do not promote an architecture with unstable Greeks unless the change explicitly targets research diagnostics and is not a production default.

## Current Limitation

The available local option dataset currently has one quote timestamp. Architecture results are useful for smoke validation and relative comparison across expirations, but they are not enough to prove temporal generalization. The data branch must collect multiple daily snapshots before architecture changes can be considered robust in a future-date sense.

