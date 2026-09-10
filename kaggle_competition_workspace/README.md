# Kaggle Playground Series S6E9 Modeling Workspace

This folder archives the modeling work for the Kaggle Playground Series S6E9 EV purchase prediction task. The competition target is `Will_Buy_EV`, and the evaluation metric is ROC AUC.

The active portfolio application lives in `../ev_adoption_platform`. This workspace keeps the experiment code, out-of-fold validation history, submissions, and notebooks that produced the model scores used by the platform.

## Best Five Approaches So Far

Only the TabM result has a documented public leaderboard score in this repository. The other scores below are five-fold out-of-fold ROC AUC values calculated from the saved OOF prediction files, which are the most reliable local score records preserved in the codebase.

| Rank | Approach | Best recorded score | Score type | Main artifact |
| --- | --- | ---: | --- | --- |
| 1 | TabM rank-average submission | **0.94480** | Public leaderboard ROC AUC | `notebooks/tabm_colab_advanced_variants.ipynb` |
| 2 | Chris-feature ensemble plus tuned LightGBM | **0.942260** | 5-fold OOF ROC AUC | `scripts/lgbm_chris_tuned_cv_blend.py` |
| 3 | Tuned LightGBM with Chris features | 0.942226 | 5-fold OOF ROC AUC | `scripts/lgbm_chris_tuned_cv_blend.py` |
| 4 | Weighted current-best plus tuned LightGBM | 0.942225 | 5-fold OOF ROC AUC | `scripts/lgbm_chris_tuned_cv_blend.py` |
| 5 | Chris XGBoost trio + LightGBM + native LightGBM blend | 0.942178 | 5-fold OOF ROC AUC | `scripts/chris_plus_lgbm_blends.py` |

## Approach Notes

### 1. TabM Rank-Average Submission

TabM was the strongest submitted approach and produced the best documented public leaderboard result: `0.94480`. It added a neural tabular model family to the project, giving the portfolio a stronger final competition score than the earlier tree-only experiments.

Relevant files:

- `notebooks/tabm_colab_baseline.ipynb`
- `notebooks/tabm_colab_advanced_variants.ipynb`

### 2. Chris-Feature Ensemble Plus Tuned LightGBM

The best saved local validation approach combined the current Chris-feature ensemble with the tuned Chris-feature LightGBM predictions. It reached `0.942260` OOF ROC AUC, making it the strongest non-TabM score preserved in the workspace.

Relevant files:

- `scripts/lgbm_chris_tuned_cv_blend.py`
- `oof_predictions/oof_lgbm_chris_tuned.csv`
- `oof_predictions/oof_chris_xgb_three_models.csv`
- `submissions/top_10/submission_chris_lgbm_tuned_blend.csv`

### 3. Tuned LightGBM With Chris Features

The tuned LightGBM model used engineered features inspired by Chris Deotte-style competition recipes, including charging access, income/subsidy interactions, environmental concern, range anxiety, and a hand-built recipe score. Its saved OOF predictions reached `0.942226` ROC AUC.

Relevant files:

- `scripts/lgbm_chris_tuned_cv_blend.py`
- `scripts/lgbm_chris_tuning_search.py`
- `configs_and_results/lgbm_chris_best_config.json`
- `oof_predictions/oof_lgbm_chris_tuned.csv`

### 4. Weighted Current-Best Plus Tuned LightGBM

This variant blended the current best Chris-feature ensemble with the tuned LightGBM using an 80/20 weighting. It reached `0.942225` OOF ROC AUC, essentially tying the tuned LightGBM while adding a little ensemble smoothing.

Relevant files:

- `scripts/lgbm_chris_tuned_cv_blend.py`
- `submissions/top_10/submission_chris_lgbm_tuned_blend.csv`

### 5. Chris XGBoost Trio + LightGBM + Native LightGBM Blend

This blend combined three XGBoost variants with Chris-style features, the Chris-feature LightGBM, and the earlier native LightGBM baseline. It reached `0.942178` OOF ROC AUC and became the foundation for the later tuned-LightGBM blend.

Relevant files:

- `scripts/chris_plus_lgbm_blends.py`
- `oof_predictions/oof_chris_xgb_three_models.csv`
- `oof_predictions/oof_lgbm_chris_features.csv`
- `submissions/top_10/submission_chris_plus_lgbm_blend.csv`

## Folder Layout

- `data/`: local competition CSVs and original EV adoption reference data
- `notebooks/`: Colab TabM notebooks and reference notebooks
- `scripts/`: training, validation, tuning, target-encoding, and blending scripts
- `oof_predictions/`: generated out-of-fold predictions used for validation and ensembling
- `configs_and_results/`: compact tuning and blend result summaries
- `submissions/top_10/`: retained high-value submission files
- `docs/project_writeup.md`: detailed chronological experiment notes

## Reproducibility Notes

Most local comparisons use stratified five-fold validation with ROC AUC. External reference rows were never placed in validation folds, so the original-data experiments were evaluated without direct leakage into competition validation rows.

Generated data files, out-of-fold predictions, and submissions may be excluded from version control because they are large and rebuildable. The scripts and result summaries are kept so the modeling path remains inspectable.
