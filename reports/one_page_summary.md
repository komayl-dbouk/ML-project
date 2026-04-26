# IoT Intrusion Detection - One Page Project Report

**Group Members:** Member 1: TODO | Member 2: TODO | Member 3: TODO  
**Repository:** TODO GitHub link | **Docker Hub Image:** TODO Docker Hub link

## Data Source and Problem

This project uses the **CICIoT2023** dataset, a labeled IoT network-traffic dataset containing benign traffic and many attack categories, including DDoS, DoS, Mirai, spoofing, reconnaissance, brute force, malware, and web attacks. The task is a **multi-class intrusion detection problem**: given network-flow features, the system predicts the traffic class and provides an alert level for security review.

The main difficulty is **class imbalance**. Some attack types appear very frequently, while rare attacks such as `Backdoor_Malware`, `BrowserHijacking`, `CommandInjection`, `DictionaryBruteForce`, `Recon-PingSweep`, `SqlInjection`, `Uploading_Attack`, and `XSS` have much lower support. Because of this, plain accuracy can be misleading: a model can score high accuracy by performing well on common classes while still being weak on rare attacks.

## Approach and Methods

We built a supervised ML pipeline using a fixed schema of **40 network-flow features**. The project includes preprocessing, model training, model comparison, inference, evaluation, and deployment. We compared four models:

- **Logistic Regression:** simple linear baseline used to verify whether the features contain predictive signal.
- **Random Forest:** tree ensemble baseline that performs well on common classes and is conservative on rare predictions.
- **Improved GPU LightGBM:** flat multi-class classifier trained with stronger regularization to reduce overfitting.
- **Hierarchical LightGBM:** final model. It predicts the attack family first, then predicts the subtype inside that family.

To avoid misleading results, we evaluated with **Accuracy, Macro Precision, Macro-F1, Weighted-F1, rare-class Precision/Recall/F1, and AUPRC when available**. We also checked duplicate overlap between archive training samples and the official test set, removed overlapping rows during diagnostic training, and used shuffled-label sanity checks to verify the models were learning real signal rather than a broken pipeline.

## Final Results on Official Test Set

| Model | Accuracy | Macro Precision | Macro-F1 | Weighted-F1 | Rare Precision | Rare Recall | Rare F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7688 | 0.4621 | 0.4405 | 0.7330 | 0.0490 | 0.1980 | 0.0519 |
| Random Forest | 0.9876 | 0.7737 | 0.7192 | 0.9887 | 0.4843 | 0.1819 | 0.1752 |
| Improved GPU LightGBM | 0.9890 | 0.7310 | 0.7509 | 0.9908 | 0.1100 | 0.6936 | 0.1844 |
| **Hierarchical LightGBM** | **0.9934** | **0.8243** | **0.8171** | **0.9944** | **0.3514** | **0.6408** | **0.3660** |

The **Hierarchical LightGBM** was selected as the final model because it achieved the strongest overall balance: best accuracy, best macro precision, best macro-F1, best weighted-F1, and best rare-class F1. The flat improved LightGBM had strong rare recall, but its rare precision was weaker, meaning it produced more false rare-attack alerts. The hierarchical model gave a better balance between catching rare attacks and avoiding false rare alerts.

## System and Deployment

The final system is deployed through a **Streamlit dashboard**. Users can upload CSV network traffic data and receive predictions with:

- predicted class and attack family,
- confidence score,
- alert level (`INFO`, `REVIEW`, `LOW_ALERT_REVIEW`, `MEDIUM_ALERT`, `HIGH_ALERT`),
- uncertainty flag,
- top-3 predicted classes for analyst review.

The project also includes a `Dockerfile`, `.dockerignore`, and README instructions to build and run the system in Docker. The Docker image contains the working dashboard, source code, deployed models, demo data, and report files.

## Conclusion

The project shows that IoT intrusion detection requires more than high accuracy because the dataset is imbalanced. By comparing baselines, improving LightGBM regularization, checking overfitting/leakage risks, and selecting a hierarchical family-to-subtype model, the final system provides a stronger and more explainable intrusion detection pipeline. Confidence, alert level, and top-3 predictions are included to make uncertain and rare-class cases reviewable instead of blindly trusting a single class label.
