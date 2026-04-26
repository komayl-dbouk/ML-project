# IoT Intrusion Detection - Detailed Presentation Outline

Use this file to build the final slides in PowerPoint, Google Slides, Canva, or Figma. Each slide includes suggested content and speaker notes.

---

## Slide 1 - Title

**IoT Intrusion Detection Using Machine Learning**

**Subtitle:** Multi-class attack detection with hierarchical LightGBM and Streamlit deployment

**Group Members:** TODO, TODO, TODO  
**Course/Instructor:** TODO  
**Date:** TODO

**Speaker notes:**  
Introduce the project as an end-to-end intrusion detection system for IoT network traffic. Mention that the work includes data preparation, model comparison, evaluation, dashboard deployment, Docker packaging, and GitHub delivery.

---

## Slide 2 - Project Goal

**Goal:** Build a working ML system that classifies IoT network traffic into benign or attack classes.

**What the system does:**
- Accepts CSV network-flow data.
- Applies the same feature schema used during training.
- Predicts the attack type.
- Reports confidence and alert level.
- Shows top-3 possible classes for uncertain cases.

**Speaker notes:**  
Emphasize that this is not only a notebook experiment. The project became a usable dashboard where a user can upload traffic data and get predictions.

---

## Slide 3 - Data Source

**Dataset:** CICIoT2023

**Traffic categories include:**
- Benign traffic
- DDoS and DoS attacks
- Mirai attacks
- Spoofing attacks
- Reconnaissance attacks
- Web attacks
- Malware and brute-force attacks

**Input representation:**  
Network-flow features such as packet rates, protocol indicators, flag counts, packet-size statistics, timing, variance, and covariance features.

**Speaker notes:**  
Explain that the dataset is useful because it covers many IoT attack behaviors, but this also makes the classification task harder: many classes are similar or heavily imbalanced.

---

## Slide 4 - Main Challenge: Class Imbalance

**Problem:** The dataset is highly imbalanced.

**Why this matters:**
- Common flood attacks dominate the data.
- Rare attacks have much fewer examples.
- Accuracy can look excellent even when rare classes are weak.
- In cybersecurity, rare attacks are often important.

**Rare classes used in evaluation:**
- `Backdoor_Malware`
- `BrowserHijacking`
- `CommandInjection`
- `DictionaryBruteForce`
- `Recon-PingSweep`
- `SqlInjection`
- `Uploading_Attack`
- `XSS`

**Speaker notes:**  
This slide is important. Say clearly that the team expected imbalance and therefore did not rely only on accuracy. This makes the evaluation more professional.

---

## Slide 5 - Why Accuracy Alone Is Misleading

**Example issue:**

A model can have very high weighted-F1 because it predicts common classes well, but still perform poorly on rare attacks.

**Metrics used instead:**
- Accuracy: overall correctness
- Macro Precision: average precision across classes
- Macro-F1: treats all classes more equally
- Weighted-F1: reflects dataset distribution
- Rare Precision: reliability of rare-attack alerts
- Rare Recall: ability to catch rare attacks
- Rare F1: balance between rare precision and recall
- AUPRC: useful for imbalanced classification when probabilities are available

**Speaker notes:**  
Explain macro-F1 in simple language: each class gets equal importance, so rare classes cannot disappear behind the common classes.

---

## Slide 6 - Preprocessing and Feature Pipeline

**Pipeline steps:**
- Load CSV data.
- Drop unused or inconsistent columns.
- Keep a fixed list of 40 expected features.
- Preserve feature order for training and inference.
- Use the same schema in the dashboard inference engine.

**Why this matters:**
- Prevents train/inference mismatch.
- Makes uploaded files compatible with the saved models.
- Keeps the dashboard stable.

**Speaker notes:**  
Mention that keeping the exact feature order is critical for ML deployment. If columns change between training and inference, predictions can be wrong even if the model file loads.

---

## Slide 7 - Models Compared

**1. Logistic Regression**
- Simple linear baseline.
- Useful to verify basic signal.
- Not strong enough for final deployment.

**2. Random Forest**
- Tree ensemble baseline.
- Conservative on rare attacks.
- Higher rare precision but low rare recall.

**3. Improved GPU LightGBM**
- Flat multi-class model.
- Faster/stronger gradient boosting model.
- Regularized to reduce overfitting.

**4. Hierarchical LightGBM**
- Stage 1: predict attack family.
- Stage 2: predict subtype inside the family.
- Selected as final model.

**Speaker notes:**  
Frame the models as an evolution from simple baseline to final architecture. This shows the team did not randomly pick one model.

---

## Slide 8 - Improvement and Overfitting Work

**Problem observed:**  
The first flat LightGBM showed a train-validation gap, meaning overfitting risk.

**Improvement applied:**
- Smaller trees
- Lower learning rate
- Stronger L1/L2 regularization
- More restrictive leaf settings
- Subsampling and column sampling
- GPU training for faster experimentation

**Observed effect:**
- Train-validation gap reduced.
- Official test Macro-F1 improved.
- Rare recall improved for the flat model.

**Speaker notes:**  
Be honest here: the team improved the flat LightGBM, then selected the hierarchical model because it performed best overall. That is a valid engineering decision.

---

## Slide 9 - Leakage and Sanity Checks

**Checks performed:**
- Duplicate/overlap check between archive sample and official test set.
- De-overlap filtering during diagnostic training.
- Shuffled-label sanity test.

**Why this matters:**
- Prevents inflated results from duplicate rows.
- Confirms the model is learning real signal.
- Helps detect broken pipelines or accidental leakage.

**Speaker notes:**  
This slide makes the work look much more serious. Many student projects skip leakage checks; mention that this was done because high accuracy in intrusion datasets can be suspicious.

---

## Slide 10 - Official Test Results

| Model | Accuracy | Macro Precision | Macro-F1 | Weighted-F1 | Rare Precision | Rare Recall | Rare F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7688 | 0.4621 | 0.4405 | 0.7330 | 0.0490 | 0.1980 | 0.0519 |
| Random Forest | 0.9876 | 0.7737 | 0.7192 | 0.9887 | 0.4843 | 0.1819 | 0.1752 |
| Improved GPU LightGBM | 0.9890 | 0.7310 | 0.7509 | 0.9908 | 0.1100 | 0.6936 | 0.1844 |
| **Hierarchical LightGBM** | **0.9934** | **0.8243** | **0.8171** | **0.9944** | **0.3514** | **0.6408** | **0.3660** |

**Speaker notes:**  
Do not only say “accuracy is high.” Say the selected model has the best Macro-F1 and rare F1, which are more important for imbalanced multi-class intrusion detection.

---

## Slide 11 - Model Interpretation

**Logistic Regression**
- Underfits the problem.
- Too simple for complex attack patterns.

**Random Forest**
- Conservative rare-attack behavior.
- Fewer rare false positives, but misses many rare attacks.

**Improved GPU LightGBM**
- Strong flat model.
- Good rare recall and AUPRC.
- Rare precision still weak.

**Hierarchical LightGBM**
- Best overall balance.
- Strongest macro-F1 and rare F1.
- Final selected model.

**Speaker notes:**  
This is where you show you understand tradeoffs. Random Forest is not useless; it has a different behavior. LightGBM catches more rare attacks but produces more false rare alerts. Hierarchical gives the best balance.

---

## Slide 12 - Final Architecture

**Flow:**

CSV upload -> preprocessing -> hierarchical model -> prediction output -> dashboard visualization

**Prediction output includes:**
- Predicted family
- Predicted class
- Confidence
- Alert level
- Uncertainty flag
- Top-3 class alternatives

**Speaker notes:**  
Explain top-3 as an analyst support feature, especially for rare attacks. The system should not hide uncertainty.

---

## Slide 13 - Alert Levels and Confidence

**Alert logic:**
- `INFO`: high-confidence benign traffic
- `REVIEW`: benign prediction with lower confidence
- `HIGH_ALERT`: attack prediction with high confidence
- `MEDIUM_ALERT`: attack prediction with medium confidence
- `LOW_ALERT_REVIEW`: attack prediction with low confidence

**Why confidence matters:**
- A predicted class alone can be misleading.
- Low-confidence attacks should be reviewed.
- Top-3 classes help explain model uncertainty.

**Speaker notes:**  
Make clear that low alert does not mean safe. It means attack predicted, but confidence is low, so analyst review is needed.

---

## Slide 14 - Dashboard

**Dashboard features:**
- CSV upload
- Chunk processing for larger files
- Model selection
- Manual single-sample simulation
- Result table
- Alert distribution chart
- Top attack-type chart
- Downloadable prediction CSV

**Speaker notes:**  
This slide should include screenshots in the final presentation if possible. Mention that the dashboard uses the same inference engine as the saved model pipeline.

---

## Slide 15 - Docker and Deployment

**Deliverables included:**
- `Dockerfile`
- `.dockerignore`
- Public Docker Hub image
- README run instructions
- GitHub repository
- One-page report
- Presentation slides

**Docker run command:**

```bash
docker run --rm -p 8501:8501 YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest
```

**Speaker notes:**  
Say that Docker makes the project easier to run on another machine without manually recreating the environment.

---

## Slide 16 - Team Contributions

**Member 1 - ML and Evaluation**
- Model training and comparison
- Overfitting diagnostics
- Metrics and result table

**Member 2 - Inference and Dashboard**
- Prediction engine
- Streamlit dashboard
- Alert level and top-3 outputs

**Member 3 - Deployment and Documentation**
- Dockerfile
- README
- Report and presentation
- GitHub/Docker Hub packaging

**Speaker notes:**  
Adjust this slide to match real team responsibilities. The instructor asked to see commits, so connect each contribution to files in the GitHub repo.

---

## Slide 17 - Limitations

**Current limitations:**
- Dataset imbalance still affects rare classes.
- Some rare attacks remain harder to classify.
- Confidence is useful but not a perfect probability guarantee.
- Results depend on the dataset distribution.
- Real production IDS would require live traffic validation and drift monitoring.

**Speaker notes:**  
This shows maturity. Do not claim the model is perfect. Claim it is evaluated carefully and designed to expose uncertainty.

---

## Slide 18 - Conclusion

**Final decision:**  
The **Hierarchical LightGBM** model is selected as the final deployed model.

**Why:**
- Best official-test accuracy.
- Best macro precision.
- Best macro-F1.
- Best rare-class F1.
- More balanced than the flat models.

**Final takeaway:**  
For imbalanced IoT intrusion detection, high accuracy is not enough. A responsible system should report macro metrics, rare-class behavior, confidence, alert level, and uncertainty.

**Speaker notes:**  
End with the main message: the project is not only about getting a high score. It is about building an IDS pipeline that gives useful and reviewable predictions.
