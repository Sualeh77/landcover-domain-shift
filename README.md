# Iowa to Sahel: Land-Cover Domain Shift

A segmentation model trained on Iowa Sentinel-2 chips (0.56 mIoU) drops to 0.06 on Niger. I diagnose why (covariate + prior + concept shift), try a ladder of fixes, and find that labeling just 30 target chips and fine-tuning gets you most of the way there (0.32 mIoU, +26 pts).

| Method | Source | Target |
|--------|--------|--------|
| Baseline | 0.561 | 0.063 |
| + photometric aug | 0.520 | 0.129 |
| + self-training | 0.522 | 0.192 |
| **+ 30 active labels** | **0.515** | **0.322** |
| Oracle (all 240 labels) | 0.522 | 0.349 |

Full writeup in `report/report.md`.

## Repo structure

```
notebooks/   01_data  02_baseline  03_diagnostics  04_interventions
src/         datasets, models, training, diagnostics, interventions
data/        sentinel-2 chips (not committed, downloaded via GEE)
checkpoints/ model weights (not committed)
report/      report.md + figures
documents/   notes, plan
```

## Setup

Need Python 3.11, [uv](https://docs.astral.sh/uv/), and a GEE account.

```bash
uv sync --extra notebooks
uv run python -m ipykernel install --user --name landcover --display-name "Python 3.11 (landcover)"
uv run earthengine authenticate
```

Set your GEE project ID in `src/data_acquisition.py` (default is `landcover-shift`).

On macOS Apple Silicon: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

## Running

Run notebooks 01-04 in order. Notebook 01 downloads data from GEE (~5 min). Notebooks 02-04 do training (~40 min, ~15 min, ~2.5 hrs respectively on MPS). Data and checkpoints aren't in git.

Notebook 04 was run in pieces during development, so its inline outputs aren't saved. The final results are in `checkpoints/intervention_results.json` and the figure is at `report/figures/04_results_ladder.png`.
