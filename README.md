# Streaming IoT Intrusion Detection

This project detects malicious IoT network traffic using the CICIoT2023 dataset. It includes preprocessing, model training/evaluation scripts, saved machine learning models, and a Streamlit dashboard for running intrusion detection on uploaded CSV files.

## Final Model

The default deployed model is:

```text
Hierarchical Family + Subtype
```

This model predicts the attack family first, then predicts the subtype inside that family. It was selected because it gave the best official-test balance between overall performance and rare-attack detection.

## Official Test Summary

| Model | Accuracy | Macro Precision | Macro-F1 | Weighted-F1 | Rare Precision | Rare Recall | Rare F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7688 | 0.4621 | 0.4405 | 0.7330 | 0.0490 | 0.1980 | 0.0519 |
| Random Forest | 0.9876 | 0.7737 | 0.7192 | 0.9887 | 0.4843 | 0.1819 | 0.1752 |
| Improved GPU LightGBM | 0.9890 | 0.7310 | 0.7509 | 0.9908 | 0.1100 | 0.6936 | 0.1844 |
| Hierarchical LightGBM | 0.9934 | 0.8243 | 0.8171 | 0.9944 | 0.3514 | 0.6408 | 0.3660 |

## Run Locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .MLvenv
.\.MLvenv\Scripts\activate
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run dashboard/app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Run With Docker

Build the Docker image:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest .
```

Run the container:

```bash
docker run --rm -p 8501:8501 YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest
```

Open:

```text
http://localhost:8501
```

## Docker Hub Image

After building and testing locally, push the image:

```bash
docker login
docker push YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest
```

Public Docker Hub image:

```text
TODO: https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection
```

## Project Structure

```text
dashboard/   Streamlit dashboard
src/         training, evaluation, inference, and utility scripts
models/      deployed saved models
data/demo/   small demo CSV for testing
reports/     one-page report and presentation slides outline
tests/       project tests
```

## Important Notes

- The full raw CICIoT2023 training archive is intentionally excluded from the Docker image because it is very large.
- The Docker image contains the code, dashboard, demo data, reports, and deployed model files needed to run inference.
- The dashboard reports predicted class, confidence, alert level, uncertainty, and top-3 predicted classes.
- Docker uses `requirements-docker.txt`, a smaller pinned runtime dependency file. The full `requirements.txt` is kept for local development and notebooks.

## Reproduce Evaluation

Evaluate a saved model on the official test CSV:

```powershell
python src\evaluate_saved_lightgbm_fast.py --model-path models\hierarchical_lightgbm_bundle.joblib
```

Train the hierarchical model:

```powershell
python src\train_hierarchical_model.py --deoverlap-official-test
```

Train the improved flat GPU LightGBM:

```powershell
python src\train_compare_on_archive.py --skip-cpu-models --sanity-checks --deoverlap-official-test --skip-reports --lgbm-device gpu --lgbm-preset lgbm_stable_strong
```

Use `--lgbm-device cpu` only if GPU LightGBM is not available.
