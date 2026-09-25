# Iowa → Sahel: Diagnosing and Fixing Land-Cover Segmentation Domain Shift

Interview assignment: a land-cover segmentation model scores 0.85 mIoU on temperate North America and 0.41 mIoU on semi-arid chips from a different season (same sensor). Diagnose why, and propose the intervention to bet on.

## Layout

```
documents/   plan, notes, decision log
notebooks/   01_data, 02_baseline, 03_diagnostics, 04_interventions
src/         reusable code (datasets, models, augmentations, metrics)
data/        source/ and target/ chips (git-ignored)
report/      report.md + figures/
```

See `documents/12_hour_plan.md` for the execution plan.
