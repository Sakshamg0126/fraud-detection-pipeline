# Fraud Detection Pipeline

End-to-end fraud detection on the ULB Credit Card Fraud dataset, built with
reproducible training, experiment tracking, and tests.

## Highlights
- Time-based train/validation/test split (no leakage from the future)
- XGBoost with class-imbalance handling, evaluated with PR-AUC
- Cost-based decision threshold chosen on validation data
- Config-driven training, MLflow tracking, pytest tests

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download `creditcard.csv` from Kaggle (`mlg-ulb/creditcardfraud`) into `data/raw/`.

## Usage
```bash
python -m src.train     # train and log to MLflow
python -m pytest -v     # run tests
mlflow ui               # view runs at http://localhost:5000
```

## Results
(Fill in after running `python -m src.train`.)

## Roadmap
- [x] Baseline model and cost-based threshold
- [x] Tests
- [ ] FastAPI serving and Docker
- [ ] GitHub Actions CI, DVC, drift monitoring