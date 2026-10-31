# BreatheSmart: Predicting Air Quality and Health Risk Using Machine Learning

BreatheSmart is an educational machine-learning prototype that predicts an air-quality category and a pollution-exposure risk category from pollutant and weather measurements. It includes a Streamlit dashboard, sample CSV, model training, and evaluation.

## Important data and safety note

The included `data/breathe_smart_sample.csv` is **synthetic demonstration data**. Its labels are generated from simplified pollutant-concentration rules. The model metrics therefore demonstrate a working ML pipeline only; they do not establish real-world predictive accuracy. The app is not a medical device and does not diagnose illness or estimate an individual's clinical probability of harm.

For a serious deployment, replace the sample with quality-controlled observations from a trusted monitoring source, document pollutant units and averaging periods, use a region-appropriate official AQI calculation, and evaluate against a geographically and temporally separate real dataset.

## Features

- Interactive pollutant inputs for PM2.5, PM10, NO2, SO2, CO and O3
- Temperature and relative humidity inputs
- Models compared: Random Forest, Extra Trees, Decision Tree and Logistic Regression
- Saved best models for AQI category and exposure-risk category
- Accuracy and macro-F1 metrics on a held-out synthetic test set
- Streamlit charts and practical, non-diagnostic guidance

## Folder structure

```text
BreatheSmart/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── breathe_smart_sample.csv
└── models/
    ├── aqi_model.joblib
    ├── health_model.joblib
    └── metrics.json
```

## Run on Windows

Open a terminal in the BreatheSmart folder.

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### If using Anaconda

```bash
conda create -n breathesmart python=3.11 -y
conda activate breathesmart
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

## Workflow

1. Generate the sample dataset.
2. Split it into training and testing subsets.
3. Train four candidate classifiers for each target.
4. Select the model with the highest macro-F1 score.
5. Save the selected models and evaluation metrics.
6. Run the dashboard and enter pollutant readings.

## Suggested real-data extension

- Obtain measured observations from CPCB or another trusted monitoring provider.
- Keep source, station, date/time, units and averaging periods with every observation.
- Prevent data leakage by splitting by time or monitoring station, rather than randomly mixing near-duplicate readings.
- Calculate official AQI according to the applicable regional standard when the required pollutants and averaging periods are available.
- Validate predictions on real held-out data and report per-class precision, recall, F1, confusion matrix and calibration.
- Add live data only through a documented, reliable source/API and show the reading timestamp.

## References

- CPCB National AQI: https://app.cpcbccr.com/AQIIndia
- CPCB AQI information: https://app.cpcbccr.com/ccr_docs/About_AQI.pdf
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
