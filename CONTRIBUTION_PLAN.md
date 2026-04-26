# Three-Person GitHub Contribution Plan

The instructor wants visible GitHub work from every team member. Use separate branches or commit directly in sequence if your team is small and coordinated.

## Member 1 - ML and Evaluation

Suggested ownership:
- `src/train_compare_on_archive.py`
- `src/train_hierarchical_model.py`
- `src/evaluate_saved_lightgbm_fast.py`
- `reports/one_page_summary.md` results table

Suggested commits:

```bash
git add src/train_compare_on_archive.py src/train_hierarchical_model.py src/evaluate_saved_lightgbm_fast.py
git commit -m "Add model evaluation and overfitting diagnostics"

git add reports/one_page_summary.md
git commit -m "Document final model results"
```

## Member 2 - Dashboard and Inference

Suggested ownership:
- `src/inference_engine.py`
- `dashboard/app.py`
- `models/` deployed model references

Suggested commits:

```bash
git add src/inference_engine.py dashboard/app.py
git commit -m "Set hierarchical model as default inference pipeline"

git add dashboard/app.py
git commit -m "Improve dashboard model selection guidance"
```

## Member 3 - Docker, README, and Presentation

Suggested ownership:
- `Dockerfile`
- `.dockerignore`
- `README.md`
- `reports/presentation_slides.md`

Suggested commits:

```bash
git add Dockerfile .dockerignore README.md
git commit -m "Add Docker deployment instructions"

git add reports/presentation_slides.md
git commit -m "Add project presentation slides outline"
```

## Final Docker Hub Steps

After Docker Desktop is running:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest .
docker run --rm -p 8501:8501 YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest
docker login
docker push YOUR_DOCKERHUB_USERNAME/iot-intrusion-detection:latest
```

Update `README.md` and `reports/one_page_summary.md` with:
- real group member names
- public Docker Hub link
- GitHub repository link if required

## Recommended Presentation Story

1. The dataset is imbalanced, so accuracy alone is misleading.
2. Baseline models were trained first.
3. LightGBM improved the flat multiclass approach.
4. The hierarchical family -> subtype model gave the best final balance.
5. The dashboard shows prediction, confidence, alert level, and top-3 alternatives for review.
