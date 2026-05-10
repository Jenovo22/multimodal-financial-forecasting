# FINN Architecture Validation

This document defines the current validation protocol for the `experiment/model-architecture` branch.

## Goal

Validate FINN architecture changes without changing defaults blindly. Every architecture experiment must compare against Black-Scholes-Merton on the same train/validation/test split and must inspect basic Greek/no-arbitrage diagnostics.

## Current Recommended Baseline

The current baseline architecture is:

```text
prediction_mode = bsm_residual
hidden_dims = 32,32
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

This is safer than direct price learning while the dataset is small. The
`32,32` default is intentionally compact because it outperformed wider networks
in the current expiration-split validation and kept Greek diagnostics clean.

## Experiment Matrix

The architecture validation runner currently covers:

| Experiment | Mode | Hidden Dims | Activation | Residual Scale | Experts |
| --- | --- | --- | --- | ---: | ---: |
| `baseline_64x64_silu_scale050` | `bsm_residual` | `64,64` | `silu` | `0.50` | `1` |
| `compact_32x32_silu_scale050` | `bsm_residual` | `32,32` | `silu` | `0.50` | `1` |
| `mixture_32x32_silu_3experts_scale050` | `bsm_residual_mixture` | `32,32` | `silu` | `0.50` | `3` |
| `wide_128x128_silu_scale050` | `bsm_residual` | `128,128` | `silu` | `0.50` | `1` |
| `baseline_64x64_gelu_scale050` | `bsm_residual` | `64,64` | `gelu` | `0.50` | `1` |
| `conservative_64x64_silu_scale025` | `bsm_residual` | `64,64` | `silu` | `0.25` | `1` |
| `flexible_64x64_silu_scale075` | `bsm_residual` | `64,64` | `silu` | `0.75` | `1` |

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

## Latest Local Validation

Run date: 2026-05-08.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_finn_architecture_validation.py `
  --epochs 300 `
  --output-dir $env:TEMP\finn_architecture_validation_full
```

Results:

| Rank | Experiment | Final MAE | Final RMSE | BSM MAE | Gamma Negative |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `compact_32x32_silu_scale050` | `0.181766` | `0.213050` | `1.493047` | `0` |
| 2 | `baseline_64x64_gelu_scale050` | `0.184492` | `0.214675` | `1.493047` | `20` |
| 3 | `wide_128x128_silu_scale050` | `0.208463` | `0.229548` | `1.493047` | `0` |
| 4 | `flexible_64x64_silu_scale075` | `0.226326` | `0.245815` | `1.493047` | `0` |
| 5 | `baseline_64x64_silu_scale050` | `0.235514` | `0.263198` | `1.493047` | `0` |
| 6 | `conservative_64x64_silu_scale025` | `0.709183` | `0.812494` | `1.493047` | `21` |

Interpretation:

- All tested residual FINN variants beat BSM on this expiration-based test.
- `compact_32x32_silu_scale050` is the best current candidate: lowest final MAE/RMSE, zero negative-gamma rows, and fewer parameters than the 64x64 baseline.
- `baseline_64x64_gelu_scale050` is close on MAE/RMSE but has 20 negative-gamma rows, so it is less attractive as a robust default.
- `conservative_64x64_silu_scale025` is too constrained in this setup.

Recommendation:

- Keep `bsm_residual` as the architecture family.
- Promote `32,32 + SiLU + residual_scale 0.50` as the current compact default.
- Re-test it with `--split-strategy timestamp` after multiple option snapshots exist before treating it as production-robust.

## Experimental Residual Mixture

Run date: 2026-05-09.

The branch also includes an experimental `bsm_residual_mixture` mode. It keeps
the BSM anchor but predicts several bounded residual experts plus softmax gate
weights:

```text
price = BSM + residual_scale * anchor_scale * sum(gate_i * tanh(residual_i))
```

This is designed to let the model learn different residual corrections across
moneyness, maturity and regime zones without abandoning the analytical anchor.

Validation against the compact default:

| Experiment | Final MAE | Final RMSE | BSM MAE | Gamma Negative |
| --- | ---: | ---: | ---: | ---: |
| `compact_32x32_silu_scale050` | `0.181766` | `0.213050` | `1.493047` | `0` |
| `mixture_32x32_silu_3experts_scale050` | `0.194962` | `0.227681` | `1.493047` | `0` |

Interpretation:

- The mixture mode is materially better than BSM and preserves clean gamma diagnostics.
- It did not beat the compact residual default on the current one-snapshot, expiration-split dataset.
- Keep it as an architecture research option, not as the default, until timestamp-split validation with more snapshots proves a benefit.
