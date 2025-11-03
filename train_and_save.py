import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Config
DATA_PATH = "Crop_recommendation.csv"
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"
LE_FILE = "le.joblib"
METRICS_FILE = "metrics.json"

print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    url = "https://raw.githubusercontent.com/insaid2018/Term-Project/master/Crop_recommendation.csv"
    df = pd.read_csv(url)

X = df[FEATURES].values
y = df['label'].values

print("Encoding labels and splitting...")
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training RandomForest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

print("Evaluating...")
preds = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
cm = confusion_matrix(y_test, preds)

print(f"Accuracy: {acc:.4f}")

# Persist artifacts
print("Saving model artifacts...")
joblib.dump(clf, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(le, LE_FILE)

metrics = {
    "accuracy": float(acc),
    "report": report,
    "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else None,
    "trained_at": datetime.utcnow().isoformat() + "Z"
}
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Done. Files saved:")
print(" -", MODEL_FILE)
print(" -", SCALER_FILE)
print(" -", LE_FILE)
print(" -", METRICS_FILE)
