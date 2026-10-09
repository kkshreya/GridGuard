# ⚡ GridGuard — AI-Based Electricity Theft Detection & Inspection Management

An upgraded version of the original *Electricity Theft Detection* mini-project (Flask + Keras ANN).
It keeps your original stack (Flask, SQLite, Keras ANN, same 9 input features) and turns it into a
complete, explainable, workflow-driven system.

## What's new compared with the original project

| Area | Original | GridGuard |
|---|---|---|
| Models | 1 ANN | Logistic Regression, Random Forest, XGBoost, ANN **+ soft-voting ensemble** |
| Evaluation | One train/test split | 5-fold stratified CV, ROC-AUC, PR-AUC, confusion matrix, threshold analysis, **noise robustness stress-test** |
| Unlabelled data | – | **Isolation Forest** (unsupervised anomaly score) |
| Features | 9 raw inputs | 23 engineered features (apparent / real / reactive power, energy-per-amp, consumer-type baselines…) |
| Explainability | none | **SHAP** contributions (XGBoost) + plain-English reasons for every alert |
| Output | Theft / Normal | **Risk score 0–100**, Low/Medium/High/Critical, ranked inspection list |
| Data handling | trusted blindly | Validation with row-level error report (range checks, bad categories, missing columns) |
| Workflow | – | Alerts become **inspection cases** (assign, inspect, confirm / false alarm, notes) |
| Learning | static model | **Feedback loop**: confirmed/false cases retrain the model, before/after F1 shown |
| Users | any user | Roles (admin / inspector), hashed passwords, CSRF protection, per-user API keys |
| Interface | basic forms | Dashboard with charts, bulk CSV scan, downloadable results, model-performance page |
| Integration | – | REST API `POST /api/predict` |
| Quality | – | 16 automated tests, Dockerfile, train/serve-consistent feature pipeline |

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python manage.py seed          # OPTIONAL: demo users + demo scans/cases
python app.py                  # open http://127.0.0.1:5000
```

* Pre-trained models are included. If they don't load on your machine (e.g. different library versions),
  the app **retrains automatically on first start** (~1 min).
* No `seed`? Just register — **the first account becomes the administrator**.
* After `seed`: `admin / Admin@123` and `inspector / Inspector@123` (demo only — change for real use).
* Full retraining with cross-validation and fresh plots: `python manage.py train`.
* If TensorFlow can't be installed on your Python version, delete that line in `requirements.txt`; the
  app runs with XGBoost + Random Forest and simply skips the ANN.

## 5-minute demo script (for viva / presentation)

1. **Home → Log in as admin** → *Dashboard*: KPIs, risk distribution, theft by consumer type/season, 14-day trend.
2. **Predict**: Residential, 180 kWh → *Critical*; show the SHAP bars and plain-English reason. Then 12 kWh → *Low*.
3. **Bulk scan**: download *sample CSV* → upload it → ranked inspection list, filter by *Critical*, download results.
4. **Cases**: open a case → assign it, add a note, mark *Confirmed Theft*.
5. **Models**: comparison table, ROC/PR curves, SHAP importance, and the **noise robustness chart**
   (be upfront: clean-data scores are ~100% because the dataset is synthetic — this chart shows realistic difficulty).
6. **Retrain with feedback** (admin, needs ≥20 verified cases — the seed data has 59): shows v1.0 → v1.1 with before/after F1.
7. **Profile**: show the API key and the `curl` example.

## Project structure

```
app.py            Flask app (routes, auth, CSRF, cases, API)        manage.py   train / seed / reset-db
database.py       SQLite schema                                      config.py   settings
ml/features.py    validation + feature engineering (shared by train and serve)
ml/train.py       full pipeline: CV, models, ensemble, IsolationForest, robustness, plots, retraining
ml/predictor.py   loads artifacts → risk score, anomaly score, SHAP explanation
ml/artifacts/     trained models + metrics.json
templates/ static/   UI (Bootstrap + Chart.js, bundled locally — works offline)
tests/            16 pytest tests          legacy/  your original files for reference
data/             synthetic_electricity_theft_data.csv
docs/             VIVA_QA.md, REPORT_GUIDE.md
```

## REST API

Get your key from **Profile**, then:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "X-API-Key: <your key>" -H "Content-Type: application/json" \
  -d '{"records":[{"EnergyConsumption":180,"Voltage":230,"Current":0.3,"PowerFactor":0.9,
      "AvgDemand":70,"Temperature":20,"Humidity":50,"Season":"Summer","ConsumerType":"Residential"}]}'
```

## Tests & Docker

```bash
python -m pytest tests -q
docker compose up --build        # serves on :5000, DB persisted in ./instance
```

## Honest limitations (say these before your examiner does)

* **The dataset is synthetic.** Theft rows are simply consumption above the normal range for the consumer type, so
  every model scores ~100% on clean data — same as your original project. That's why the project adds a noise stress-test
  and an unsupervised detector (ROC-AUC ≈ 0.90 without labels).
* SHAP ranks **Energy per amp** first. That is an artefact of how the synthetic data was generated, not a physical law.
* `Latitude/Longitude` in the dataset are random across the globe, so no theft-hotspot map was built. With real feeder/transformer
  coordinates this would be a natural extension.
* Each record is one reading, not a time-series per meter. With real smart-meter data, add rolling/seasonal features or an LSTM/1D-CNN.
* SQLite and the in-process model are right for a demo / single server; use PostgreSQL + a model service for production.

## Possible future work

Time-series (LSTM / 1D-CNN) on 30–90 day load profiles · transformer-level energy-balance check · geospatial hotspot map ·
email/SMS alerts · concept-drift monitoring · SHAP for the ANN · PostgreSQL + Celery for scheduled scans.
