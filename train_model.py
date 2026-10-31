from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breathe_smart_sample.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError("Dataset missing. Run: python generate_dataset.py")

df = pd.read_csv(DATA_PATH)
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity"]
X = df[FEATURES]
targets = {
    "aqi_model.joblib": "AQI_Category",
    "health_model.joblib": "Health_Risk",
}
X_train, X_test, y_train_all, y_test_all = train_test_split(
    X, df[["AQI_Category", "Health_Risk"]], test_size=0.2,
    random_state=42, stratify=df["AQI_Category"]
)

candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=2, class_weight="balanced", random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=250, max_depth=14, min_samples_leaf=2, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
}
results = {}
for model_file, target in targets.items():
    y_train, y_test = y_train_all[target], y_test_all[target]
    best_name, best_model, best_f1 = None, None, -1
    model_scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = f1_score(y_test, pred, average="macro", zero_division=0)
        model_scores[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(score), 4),
        }
        if score > best_f1:
            best_name, best_model, best_f1 = name, model, score
    joblib.dump({
        "model": best_model,
        "features": FEATURES,
        "target": target,
        "model_name": best_name,
        "classes": list(best_model.classes_) if hasattr(best_model, "classes_") else list(best_model.named_steps["classifier"].classes_),
    }, MODEL_DIR / model_file)
    results[target] = {
        "selected_model": best_name,
        "scores": model_scores,
        "classification_report": classification_report(y_test, best_model.predict(X_test), output_dict=True, zero_division=0),
    }
    print(f"\n{target}: selected {best_name}")
    print(pd.DataFrame(model_scores).T.sort_values("macro_f1", ascending=False))

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nModels and evaluation metrics saved in the models/ directory.")
print("Reminder: metrics measure performance on synthetic demonstration data only.")
