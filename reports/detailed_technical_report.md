# Detailed Technical Report - IoT Intrusion Detection

## 1. Project Overview

This project builds an end-to-end machine learning system for IoT intrusion detection using network-flow data. The goal is to classify each network-flow record as benign traffic or one of several attack types. The final system includes:

- data loading and validation,
- preprocessing with a fixed feature schema,
- multiple machine learning models,
- model tuning and overfitting checks,
- official-test evaluation,
- a Streamlit dashboard for inference,
- Docker packaging for deployment,
- project documentation and report files.

The selected final model is the **Hierarchical Family + Subtype LightGBM** model. It predicts the attack family first and then predicts the exact subtype inside that family.

---

## 2. Data Source

The project uses the **CICIoT2023** dataset. This dataset contains labeled IoT network traffic with both benign and malicious traffic classes.

The classes include attack families such as:

- DDoS attacks,
- DoS attacks,
- Mirai attacks,
- spoofing attacks,
- reconnaissance attacks,
- web attacks,
- malware,
- brute-force attacks,
- benign traffic.

The raw data is stored in CSV format. 

The official test set was kept separate for final evaluation.

## 3. Data Loading

The project loads CSV files using `pandas` for normal test/demo files and `duckdb` for large archive sampling.

Important loading scripts:

For the large archive dataset, `DuckDB` is used to stream and sample data efficiently without loading the entire archive into memory at once. This is useful because the archive contains many large CSV files.

The function `sample_archive_balanced()` in `src/train_compare_on_archive.py` reads the archive and samples up to a maximum number of rows per label. This helps training stay practical while also reducing extreme dominance by common classes.

## 4. Preprocessing

The preprocessing is code-based and happens during both training and inference. There is no final saved preprocessed dataset used by the dashboard. Instead, the project applies the same schema every time data is loaded.

The preprocessing logic is mainly defined in:
The important preprocessing steps are:

1. Drop unused or inconsistent columns:

```text
ece_flag_number
cwr_flag_number
Telnet
SMTP
IRC
DHCP
```

2. Separate the target column:

```text
label
```

3. Keep exactly 40 expected features in a fixed order.

4. Validate that all required features exist.

5. During training/evaluation, convert features to `float32` where needed.

The fixed expected feature list includes network-flow statistics such as:

- duration features,
- protocol type,
- rate features,
- TCP flag counts,
- protocol indicators such as HTTP, HTTPS, DNS, TCP, UDP, ICMP,
- packet size statistics,
- timing and variance-related features.

This fixed schema is important because ML models require the same feature order during training and inference. If uploaded CSV columns are missing or reordered, the inference engine corrects the order or raises an error if required columns are missing.

## 5. Class Imbalance Problem

The dataset is highly imbalanced. Some classes appear very frequently, especially flood-based attacks, while some attack types are rare.

Rare classes used in evaluation were:

```text
Backdoor_Malware
BrowserHijacking
CommandInjection
DictionaryBruteForce
Recon-PingSweep
SqlInjection
Uploading_Attack
XSS
```
This matters because accuracy alone can be misleading. A model can achieve very high accuracy by performing well on common classes while performing poorly on rare attacks.

Because of this, the project evaluates models using:

- Accuracy,
- Macro Precision,
- Macro-F1,
- Weighted-F1,
- Rare Precision,
- Rare Recall,
- Rare F1,
- AUPRC where probabilities are available.

Macro-F1 and rare-class metrics are especially important for this project because they reveal weaknesses hidden by overall accuracy.

## 6. Models Tested

The project compares four main model types.

### 6.1 Logistic Regression

Logistic Regression is used as a simple baseline.

Purpose:

- confirm that the features contain predictive signal,
- provide a simple comparison point,
- show the limitation of linear models on this problem.

Result:

The model underfits the problem. It is too simple for the complex relationships between network-flow features and attack classes.

### 6.2 Random Forest

Random Forest is a tree-based ensemble baseline.

Purpose:

- compare a non-linear classical ML model,
- test whether decision trees improve over a linear baseline.

Behavior:

- strong performance on common classes,
- relatively high rare precision,
- weak rare recall.

This means Random Forest is conservative: when it predicts a rare attack it is more likely to be correct, but it misses many rare attacks.

### 6.3 Improved GPU LightGBM

LightGBM is a gradient boosting model and is the strongest flat multi-class classifier in this project.

Purpose:

- improve performance over Logistic Regression and Random Forest,
- use GPU training for faster experimentation,
- reduce overfitting with a more regularized preset.

The improved flat LightGBM uses stronger regularization compared with the first overfitting-prone version:

- smaller tree depth,
- fewer leaves,
- lower learning rate,
- higher `min_child_samples`,
- L1 and L2 regularization,
- row subsampling,
- column subsampling,
- early stopping.

This reduced the train-validation gap and improved generalization compared with the earlier flat model.

### 6.4 Hierarchical Family + Subtype LightGBM

The final selected model is hierarchical.

It works in two stages:

1. Predict the broad attack family.
2. Predict the subtype inside that family.

Example:

```text
Input traffic -> Family model predicts "WebAttack"
              -> WebAttack subtype model predicts "SqlInjection"
```

Why this helps:

- many classes are related inside families,
- rare subtypes are easier to handle after reducing the search space,
- it improves balance between common and rare classes,
- it achieved the best official-test Macro-F1 and rare F1.

---

## 7. Tuning and Improvement Work

The main tuning work focused on reducing overfitting and improving generalization.

For the flat GPU LightGBM, a more regularized preset was added:

```text
lgbm_stable_strong
```

The regularized preset uses:

- lower `num_leaves`,
- lower `max_depth`,
- higher `min_child_samples`,
- lower `learning_rate`,
- `reg_alpha`,
- `reg_lambda`,
- `subsample`,
- `colsample_bytree`,
- `min_split_gain`.

The goal was to prevent the model from memorizing training patterns and improve validation/test behavior.

Observed improvement:

```text
Old flat LightGBM Train-Val gap: 0.1289
Improved flat LightGBM Train-Val gap: 0.0665
```

So the improved flat model reduced overfitting significantly.

The hierarchical model was then selected as the final deployed model because it had stronger official-test performance than the flat model.

---

## 8. Leakage and Sanity Checks

High accuracy in intrusion datasets can be suspicious, so the project included checks to make sure results were not inflated.

### 8.1 Duplicate/Overlap Check

The project checks whether sampled archive training rows overlap with official test rows by hashing the expected feature columns.

If overlap exists, rows can be removed using de-overlap filtering.

Purpose:

- prevent exact duplicate feature rows from appearing in both training and test evaluation,
- make test results more honest.

### 8.2 Shuffled-Label Sanity Test

The model was also tested with shuffled labels.

Purpose:

- if labels are random, a valid model should collapse near random performance,
- if performance stays high with shuffled labels, the pipeline may have leakage or a bug.

Observed behavior:

The shuffled-label model performed near random, which supports that the real model learned actual signal.

---

## 9. Evaluation Metrics

The project uses several metrics because each one answers a different question.

### Accuracy

Overall percentage of correct predictions.

Useful, but misleading with imbalanced data.

### Macro Precision

Average precision across classes, treating each class equally.

Useful for seeing how reliable predictions are across all classes.

### Macro-F1

Average F1-score across classes, treating each class equally.

Very important for imbalanced multi-class classification.

### Weighted-F1

F1-score weighted by class support.

Useful for overall dataset performance, but can hide rare-class weakness.

### Rare Precision

Measures how reliable rare-attack predictions are.

High rare precision means fewer false rare-attack alerts.

### Rare Recall

Measures how many rare attacks are successfully caught.

High rare recall means fewer missed rare attacks.

### Rare F1

Balances rare precision and rare recall.

This is one of the most useful metrics for judging rare-class behavior.

### AUPRC

Area under the precision-recall curve.

Useful for imbalanced classification when models expose probability outputs. It was available for the flat probabilistic models. It was not directly reported for the hierarchical model because its output is staged through family and subtype models rather than one clean probability vector over all final classes.

---

## 10. Official Test Results

Final official-test results:

| Model | Accuracy | Macro Precision | Macro-F1 | Weighted-F1 | Macro AUPRC | Rare Precision | Rare Recall | Rare F1 | Rare AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7688 | 0.4621 | 0.4405 | 0.7330 | 0.4597 | 0.0490 | 0.1980 | 0.0519 | 0.0476 |
| Random Forest | 0.9876 | 0.7737 | 0.7192 | 0.9887 | 0.7921 | 0.4843 | 0.1819 | 0.1752 | 0.2876 |
| Improved GPU LightGBM | 0.9890 | 0.7310 | 0.7509 | 0.9908 | 0.8580 | 0.1100 | 0.6936 | 0.1844 | 0.5242 |
| Hierarchical LightGBM | 0.9934 | 0.8243 | 0.8171 | 0.9944 | N/A | 0.3514 | 0.6408 | 0.3660 | N/A |

---

## 11. Model Interpretation

### Logistic Regression

This model is mainly a baseline. Its performance is much lower than the tree-based models. It underfits because the attack patterns are too complex for a simple linear classifier.

### Random Forest

Random Forest performs well overall but has low rare recall. It is careful when predicting rare attacks, but it misses many of them. It is useful as a conservative baseline but not ideal as the final IDS model.

### Improved GPU LightGBM

This is the best flat multi-class model. It has strong Macro AUPRC and high rare recall. However, rare precision is weak, meaning it catches many rare attacks but also creates more false rare-attack alerts.

### Hierarchical LightGBM

This is the best final model. It has the highest accuracy, Macro Precision, Macro-F1, Weighted-F1, and rare F1. It gives the best balance between catching rare attacks and avoiding false rare alerts.

---

## 12. Final Model Selection

The final deployed model is:

```text
Hierarchical Family + Subtype
```

Reasons:

- best official-test Accuracy,
- best Macro Precision,
- best Macro-F1,
- best Weighted-F1,
- best rare-class F1,
- better balance between rare precision and rare recall,
- more explainable structure by grouping related attack classes.

The improved flat GPU LightGBM is kept as a strong comparison model.

The Logistic Regression and Random Forest models are kept as baselines.

---

## 13. Inference and Dashboard

The deployed dashboard is implemented with Streamlit:

```text
dashboard/app.py
```

The inference engine is:

```text
src/inference_engine.py
```

Dashboard features:

- upload CSV file,
- choose model,
- run detection,
- process larger CSV files in chunks,
- display predicted class,
- display confidence,
- display alert level,
- display uncertainty,
- show top-3 predicted classes,
- show alert distribution,
- show top predicted attack types,
- download full prediction results.

The dashboard uses the same preprocessing schema as training, which reduces the risk of training/inference mismatch.

---

## 14. Alert Levels

The project converts predictions and confidence into alert levels.

Alert meaning:

```text
INFO              high-confidence benign traffic
REVIEW            benign prediction with lower confidence
HIGH_ALERT        attack prediction with confidence >= 0.90
MEDIUM_ALERT      attack prediction with confidence >= 0.60 and < 0.90
LOW_ALERT_REVIEW  attack prediction with confidence < 0.60
```

Important note:

`LOW_ALERT_REVIEW` does not mean harmless. It means the model predicted an attack but with low confidence, so analyst review is needed.

---

## 15. Top-3 Predictions

The dashboard reports top-3 predicted classes.

Purpose:

- explain uncertainty,
- show whether top classes are close,
- help review rare-attack predictions,
- reveal when the model is confused between similar attack types.

For common attacks, top-1 is often much higher than top-2 and top-3.

For rare attacks, top-1, top-2, and top-3 can be closer, which signals uncertainty.

---

## 16. Docker and Deployment

The project includes Docker support:

```text
Dockerfile
.dockerignore
requirements-docker.txt
```

Docker runs the Streamlit dashboard:

```bash
docker run --rm -p 8501:8501 komayl/iot-intrusion-detection:latest
```

The Docker image is designed to include:

- source code,
- dashboard,
- saved deployed models,
- demo data,
- reports and documentation.

Large raw/archive datasets are excluded from the image because they are too large and are not required for inference.

---

## 17. Files and Responsibilities

Important files:

```text
src/inference_engine.py              final inference logic
src/hierarchical_labels.py           family/subtype mapping
src/train_hierarchical_model.py      hierarchical training
src/train_compare_on_archive.py      flat model training and diagnostics
src/evaluate_saved_lightgbm_fast.py  fast official-test evaluation
dashboard/app.py                     Streamlit dashboard
models/hierarchical_lightgbm_bundle.joblib  final deployed model
models/archive_lightgbm.joblib       improved flat comparison model
Dockerfile                           Docker deployment
README.md                            run instructions
reports/one_page_summary.md          short report
reports/presentation_slides.md       presentation outline
```

---

## 18. Reproducibility

The scripts use fixed seeds for sampling and splitting:

```text
train_compare_on_archive.py default seed = 42
train_hierarchical_model.py default seed = 1
```

This makes sampling and train-validation splitting mostly reproducible. However, GPU LightGBM may still produce very small differences because GPU training uses parallel floating-point operations.

For maximum reproducibility, CPU training can be used, but GPU training is faster.

---

## 19. Limitations

The project has some limitations:

- The dataset is highly imbalanced, so rare classes remain difficult.
- Some rare attacks have lower reliability than common attacks.
- Confidence is useful but not a perfect guarantee.
- AUPRC is not directly comparable for the hierarchical model because probabilities are staged.
- Real-world deployment would require live network validation.
- Future work should include drift monitoring, threshold calibration, and testing on new unseen network environments.

---

## 20. Conclusion

This project developed a complete IoT intrusion detection pipeline from data loading to deployment. The team compared multiple models, evaluated beyond accuracy, addressed imbalance with macro and rare-class metrics, checked leakage risks, improved flat LightGBM regularization, and selected a hierarchical LightGBM model as the final deployed model.

The final model achieved:

```text
Accuracy:        0.9934
Macro Precision: 0.8243
Macro-F1:        0.8171
Weighted-F1:     0.9944
Rare Precision:  0.3514
Rare Recall:     0.6408
Rare F1:         0.3660
```

The final system is usable through a Streamlit dashboard and can be packaged in Docker for easier deployment.

