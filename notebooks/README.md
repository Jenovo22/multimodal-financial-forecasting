# Notebooks

This folder contains exploratory notebooks. Production workflows should move into `src/` or `scripts/` once they become repeatable.

## Structure

- `legacy/`: notebooks imported from earlier project stages.

## Rules

- Keep notebooks small when possible.
- Do not store secrets in notebook outputs.
- Prefer scripts for reproducible training, scoring and reporting.
- If a notebook becomes part of the main workflow, extract the logic into tested Python modules.

