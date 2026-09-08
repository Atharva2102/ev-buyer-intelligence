# Kaggle Competition Workspace Archive

This folder contains the original Kaggle experimentation workspace after cleanup. The active portfolio project lives in `../ev_adoption_platform`.

## Layout

- `data/`: competition CSVs and the original EV adoption reference dataset.
- `notebooks/`: Colab TabM notebooks and downloaded public-reference notebooks.
- `scripts/`: historical modeling, tuning, target-encoding, and blending scripts.
- `oof_predictions/`: saved out-of-fold prediction files used for model comparison/blending.
- `configs_and_results/`: tuning configs and compact result summaries.
- `submissions/top_10/`: the ten retained submission files from the local workspace.
- `submissions/deprecated_removed_from_active_workspace/`: older or lower-value submissions removed from the active root view.

## Retained Submission Files

The retained files preserve the most useful modeling lineage:

- `submission_chris_lgbm_tuned_blend.csv`
- `submission_chris_plus_lgbm_blend.csv`
- `submission_chris_xgb_starter_reproduction.csv`
- `submission_equal_blend.csv`
- `submission_blend_best.csv`
- `submission_lgbm_chris_tuned.csv`
- `submission_lgbm_target_encoded.csv`
- `submission_lgbm_with_original.csv`
- `submission_lightgbm_5fold.csv`
- `submission_xgb_tuned_5fold.csv`

If a stronger TabM submission is copied into the repository root or this archive later, `ev_adoption_platform/etl/build_warehouse.py` will prefer it automatically.
