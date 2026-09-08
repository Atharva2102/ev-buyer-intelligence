# Kaggle Playground Series S6E9: EV Purchase Prediction

## Most Recent Update

The latest experiment used the original EV adoption dataset as a Bayesian prior for target encoding instead of adding its rows directly to training.

New script:

```text
target_encoding_with_original_prior.py
```

This experiment created strict out-of-fold target encodings for six categorical features and two categorical combinations. The original dataset supplied prior category buy-rates, while the competition training folds supplied fold-specific statistics. Validation rows never contributed to their own encodings.

Result:

```text
Native categorical LightGBM baseline OOF AUC:       0.941897
Target encoding with original prior OOF AUC:        0.941852
Delta vs baseline:                                  -0.000045
```

Conclusion:

```text
Using the original dataset as a target-encoding prior did not improve validation.
```

The output file was still generated successfully:

```text
submission_lgbm_target_encoded.csv
```

The current best validated submission remains:

```text
submission_blend_best.csv
OOF AUC: 0.941928
```

## 1. Competition Goal

The goal of this competition is to predict whether a person is likely to buy an electric vehicle.

The target column is:

```text
Will_Buy_EV
```

In the training data, this value is either:

```text
Yes
No
```

For the test data, Kaggle hides this target. Our job is to predict the probability that each test row belongs to the positive class:

```text
Will_Buy_EV = Yes
```

The submission file must contain:

```csv
id,Will_Buy_EV
668665,0.23
668666,0.78
668667,0.11
```

The value in `Will_Buy_EV` should be a probability between `0` and `1`.

## 2. Data Overview

The workspace contains three original Kaggle files:

```text
train.csv
test.csv
sample_submission.csv
```

File sizes and shapes:

```text
train.csv:              668,665 rows, 15 columns
test.csv:               286,571 rows, 14 columns
sample_submission.csv:  286,571 rows, 2 columns
```

The training file has all feature columns plus the target. The test file has the same feature columns but does not include the target.

There are no missing values in either `train.csv` or `test.csv`.

### Target Distribution

The target is imbalanced:

```text
No     551,886
Yes    116,779
```

The positive class rate is:

```text
17.46%
```

This means only about 17 out of every 100 rows are labeled as likely EV buyers.

### Numeric Features

```text
Age
Annual_Income_USD
Daily_Commute_km
Number_of_Cars_Owned
Charging_Stations_Near_Home
Charging_Stations_Near_Work
Environmental_Concern_Level
```

### Categorical Features

```text
Gender
City_Type
Current_Car_Type
Home_Charging_Possible
Subsidy_Available
Range_Anxiety_Level
```

## 3. Evaluation Metric

The competition uses ROC AUC.

ROC AUC measures how well the model ranks positive examples above negative examples. In this competition, it answers:

> If we randomly choose one actual EV buyer and one actual non-buyer, how often does the model give the EV buyer a higher probability?

Interpretation:

```text
1.0 = perfect ranking
0.5 = random guessing
0.0 = perfectly wrong ranking
```

ROC AUC does not require us to choose a hard cutoff like `0.5`. Kaggle only needs probability scores, and the metric rewards good ranking.

## 4. Feature Engineering

We started with the raw Kaggle columns and added a few simple, interpretable features.

### Engineered Numeric Features

```text
Total_Charging_Stations
```

Sum of charging stations near home and work.

```text
Charging_Station_Difference
```

Charging stations near work minus charging stations near home.

```text
Charging_per_Commute
```

Total charging stations divided by commute distance.

```text
Income_per_Car
```

Annual income divided by number of cars owned.

### Engineered Categorical Features

```text
City_Home_Charging
```

Combination of city type and whether home charging is possible.

```text
Subsidy_Range_Anxiety
```

Combination of subsidy availability and range anxiety level.

These features were added in:

```text
starter_baseline.py
```

## 5. Models and Techniques Tried

### Logistic Regression

Logistic regression was used as a simple linear baseline and later as a possible ensemble component.

Preprocessing:

```text
One-hot encoding for categorical features
Standard scaling for numeric features
```

Result:

```text
Logistic Regression 5-fold OOF ROC AUC: 0.938106
```

This was weaker than the tree-based models, but useful as a diversity check.

Files:

```text
logistic_cross_validation.py
submission_logreg_5fold.csv
oof_logreg_5fold.csv
```

### XGBoost Baseline

XGBoost was the first strong model we used because it was already installed in the environment.

Technique:

```text
Gradient boosted decision trees
One-hot encoding for categorical features
```

Initial holdout result:

```text
Validation ROC AUC: 0.941545
```

Files:

```text
starter_baseline.py
submission_xgb_baseline.csv
```

### XGBoost 5-Fold Cross-Validation

We then moved from a single 80/20 holdout split to 5-fold stratified cross-validation.

This gave a more reliable estimate of model performance.

Results:

```text
Fold 1 ROC AUC: 0.940553
Fold 2 ROC AUC: 0.941516
Fold 3 ROC AUC: 0.942849
Fold 4 ROC AUC: 0.942398
Fold 5 ROC AUC: 0.941742

Mean fold ROC AUC: 0.941812
OOF ROC AUC:       0.941804
```

Files:

```text
xgb_cross_validation.py
submission_xgb_5fold.csv
oof_xgb_5fold.csv
```

### CatBoost

CatBoost was tested because it handles categorical features natively.

CatBoost was not installed initially, so it was installed during the workflow.

We tested:

```text
Sample CatBoost benchmark
Full-data CatBoost holdout model
```

Results:

```text
CatBoost sample holdout ROC AUC: 0.939445
CatBoost full holdout ROC AUC:   0.941028
```

CatBoost was strong, but it did not beat XGBoost or LightGBM in our current setup.

Its predictions were also very similar to XGBoost predictions, so it did not add much ensemble diversity.

Files:

```text
catboost_sample_benchmark.py
catboost_holdout_submission.py
catboost_cross_validation.py
submission_catboost_holdout.csv
```

### XGBoost Tuning

We tuned XGBoost by testing several parameter configurations.

The best tuned configuration was:

```python
{
    "n_estimators": 700,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 8,
    "reg_lambda": 6.0,
    "reg_alpha": 0.2,
}
```

This made the model slightly more regularized and conservative.

Results:

```text
Original XGBoost 5-fold OOF ROC AUC: 0.941804
Tuned XGBoost 5-fold OOF ROC AUC:    0.941825
```

The improvement was very small:

```text
+0.000021
```

Files:

```text
xgb_tuning_search.py
xgb_best_config.json
xgb_tuning_results.csv
xgb_tuned_cross_validation.py
submission_xgb_tuned_5fold.csv
oof_xgb_tuned_5fold.csv
```

### LightGBM

LightGBM was added as another strong gradient boosting model.

LightGBM was not installed initially, so it was installed during the workflow.

Technique:

```text
Gradient boosted decision trees
One-hot encoding for categorical features
```

Results:

```text
Fold 1 ROC AUC: 0.940596
Fold 2 ROC AUC: 0.941581
Fold 3 ROC AUC: 0.942886
Fold 4 ROC AUC: 0.942351
Fold 5 ROC AUC: 0.941813

Mean fold ROC AUC: 0.941845
OOF ROC AUC:       0.941836
```

LightGBM became the best single model so far.

Files:

```text
lightgbm_cross_validation.py
submission_lightgbm_5fold.csv
oof_lightgbm_5fold.csv
```

### Original Dataset Augmentation Test

The Kaggle data page stated that the competition data was inspired by an original EV adoption behavior dataset. That original dataset was added locally:

```text
OG Dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv
```

The original dataset has:

```text
10,000 rows
15 columns
```

Its columns mostly match the competition data. The only ID difference is:

```text
Competition ID column: id
Original ID column:    Buyer_ID
```

The original dataset uses the same target name and labels:

```text
Will_Buy_EV = Yes/No
```

The target rate is also very close to the competition training data:

```text
Competition train Yes rate: 17.46%
Original Yes rate:          17.50%
```

However, the original dataset has missing values in a few numeric columns:

```text
Annual_Income_USD:              178
Daily_Commute_km:               181
Environmental_Concern_Level:    184
```

We tested whether the original data helps by building a new standalone script:

```text
lightgbm_with_original.py
```

This script used LightGBM with native categorical handling instead of one-hot encoding.

Validation design:

```text
5-fold StratifiedKFold on competition train rows only
Original rows added only to the training side of each fold
Validation folds contained competition rows only
Original rows never entered validation
```

This avoided leakage and gave an honest test of whether the original rows improve competition validation performance.

Results:

```text
Baseline native-categorical LightGBM OOF AUC:      0.941897
With original rows native-categorical OOF AUC:     0.941882
Delta:                                             -0.000014
```

Conclusion:

```text
Simple row augmentation with the original dataset did not help.
```

The original data was worth testing, but adding all original rows directly to every training fold slightly reduced OOF performance. The effect was tiny, but negative.

The script still produced a valid requested submission:

```text
submission_lgbm_with_original.csv
```

This file passed basic checks:

```text
Rows: 286,571
Columns: id, Will_Buy_EV
Probabilities in [0, 1]: True
```

### Original Dataset as Target-Encoding Prior

After direct row augmentation failed to help, we tested a more careful use of the original data: using it as a prior for target encoding.

New script:

```text
target_encoding_with_original_prior.py
```

Target-encoded keys:

```text
Gender
City_Type
Current_Car_Type
Home_Charging_Possible
Subsidy_Available
Range_Anxiety_Level
City_Type + Home_Charging_Possible
Subsidy_Available + Range_Anxiety_Level
```

For each key, the script computed original-dataset category statistics:

```text
original category EV buy-rate
original category count
```

Then, for the competition training data, it built strict out-of-fold target encodings. For each fold, validation rows were encoded using only the fold's training rows, smoothed toward the original dataset's category rate:

```text
(fold_sum + alpha * original_rate) / (fold_count + alpha)
```

The default smoothing value was:

```text
alpha = 20
```

Coverage was perfect for the requested keys:

```text
Gender: 3 found, 0 fallback
City_Type: 3 found, 0 fallback
Current_Car_Type: 4 found, 0 fallback
Home_Charging_Possible: 2 found, 0 fallback
Subsidy_Available: 2 found, 0 fallback
Range_Anxiety_Level: 3 found, 0 fallback
City_Type + Home_Charging_Possible: 6 found, 0 fallback
Subsidy_Available + Range_Anxiety_Level: 6 found, 0 fallback
```

Results:

```text
Fold 1 ROC AUC: 0.940589
Fold 2 ROC AUC: 0.941614
Fold 3 ROC AUC: 0.942892
Fold 4 ROC AUC: 0.942350
Fold 5 ROC AUC: 0.941859

Mean fold ROC AUC: 0.941861
OOF ROC AUC:       0.941852
Baseline OOF AUC:  0.941897
Delta vs baseline: -0.000045
```

Conclusion:

```text
Target encoding with original-dataset priors did not help.
```

The experiment was well-designed and leakage-safe, but the encoded categorical priors did not add useful signal beyond what native-categorical LightGBM already learned.

Output file:

```text
submission_lgbm_target_encoded.csv
```

## 6. Ensembling and Stacking

After training multiple models, we tested whether combining predictions improved ROC AUC.

The important rule was that we used out-of-fold predictions for validation. This avoids leakage when evaluating blends and stacks.

### First Ensemble Attempt

Models included:

```text
Tuned XGBoost
Original XGBoost
Logistic Regression
```

Best blend:

```text
60% tuned XGBoost
40% original XGBoost
0% logistic regression
```

Result:

```text
Best weighted blend OOF ROC AUC: 0.941837
```

Logistic regression did not help.

### Ensemble With LightGBM

Models included:

```text
Tuned XGBoost
Original XGBoost
Logistic Regression
LightGBM
```

Best blend:

```text
50% LightGBM
30% tuned XGBoost
20% original XGBoost
0% logistic regression
```

Result:

```text
Best weighted blend OOF ROC AUC: 0.941928
```

This is the best validated score so far.

We also tested a logistic regression stacking meta-model:

```text
Stacked meta-model OOF ROC AUC: 0.941919
```

The simple weighted blend performed slightly better than stacking.

Files:

```text
ensemble_search.py
ensemble_results.csv
submission_blend_best.csv
submission_stack_logreg_meta.csv
```

## 7. Current Best Result

The current best validated model is the weighted blend:

```text
50% LightGBM
30% tuned XGBoost
20% original XGBoost
```

Current best OOF ROC AUC:

```text
0.941928
```

Best submission file:

```text
submission_blend_best.csv
```

This is the file that should be uploaded next to Kaggle.

## 8. Summary of Scores

```text
Logistic Regression 5-fold OOF:  0.938106
CatBoost sample holdout:         0.939445
CatBoost full holdout:           0.941028
XGBoost baseline holdout:        0.941545
XGBoost 5-fold OOF:              0.941804
Tuned XGBoost 5-fold OOF:        0.941825
LightGBM 5-fold OOF:             0.941836
Native categorical LightGBM OOF: 0.941897
LightGBM with original data OOF: 0.941882
Target encoding with original prior OOF: 0.941852
XGBoost-only blend OOF:          0.941837
LightGBM + XGBoost blend OOF:    0.941928
Stacked meta-model OOF:          0.941919
```

## 9. Main Lessons So Far

1. The dataset is clean and has no missing values.
2. The target is imbalanced, with about 17.46% positive examples.
3. Tree boosting models are much stronger than logistic regression.
4. CatBoost is solid but did not beat XGBoost or LightGBM in our current setup.
5. XGBoost tuning gave only a negligible improvement.
6. LightGBM gave the most useful single-model improvement.
7. Ensembling LightGBM with XGBoost gave the best result so far.
8. Logistic regression did not help the blend.
9. LightGBM native categorical handling improved over the earlier one-hot LightGBM run.
10. Simple training-row augmentation with the original dataset did not help.
11. Using the original dataset as a Bayesian target-encoding prior also did not help.
12. Further gains probably require better feature engineering, model diversity, or leaderboard-informed iteration.

## 10. Recommended Next Steps

The next best improvements are likely to come from adding new signal rather than making tiny parameter tweaks.

Recommended next phase:

```text
More interaction features
LightGBM tuning
Rank averaging instead of probability averaging
Testing native categorical handling in LightGBM
Trying ExtraTrees or HistGradientBoosting for ensemble diversity
```

The original dataset has now been tested in two forms:

```text
Direct row augmentation
Target-encoding prior
```

Both were slightly negative in OOF validation, so the safest current choice is to keep the best LightGBM + XGBoost blend as the main submission.

The next most promising step is probably stronger feature engineering on the competition data itself, especially numeric interactions and bins:

```text
Income bins
Commute distance bins
Charging access bins
Age-income interactions
Charging access vs range anxiety
Commute distance vs range anxiety
```

The original dataset may still be useful, but probably not through direct row augmentation. Better future uses include:

```text
Lower-weight original rows
Distribution comparison between competition and original rows
Features that measure similarity to original-data patterns
```
