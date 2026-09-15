# Results policy

This directory contains only curated, compact experiment summaries that are useful for comparison or paper/reproduction work.

## Tracked

- aggregate/summary CSV tables;
- small manually curated comparison tables;
- short metadata/documentation needed to interpret those tables.

## Not tracked

The following belong under `../outputs/` or `results/raw/` and are ignored:

- per-seed predictions and histories;
- `.npz` / `.npy` arrays;
- checkpoints and model weights;
- generated figures;
- temporary/per-run CSVs;
- logs.

Do not move a file into `results/summaries/` merely to bypass `.gitignore`. Promote a result only when it is a stable aggregate worth reviewing or citing.
