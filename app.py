import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Title and description ---
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌾 AI Crop Recommendation System")
st.write("Recommend the best crop for given soil and climate conditions using a Random Forest classifier.")

# --- Load dataset (local if present, else fallback to URL) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
    except Exception:
        url = "https://raw.githubusercontent.com/insaid2018/Term-Project/master/Crop_recommendation.csv"
        df = pd.read_csv(url)
    return df

data = load_data()
st.write("### Dataset preview", data.head())

# --- Features ---
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# --- Sidebar inputs ---
st.sidebar.header("Enter your soil & climate details 🌱")
def user_inputs():
    n = st.sidebar.number_input("Nitrogen (N)", 0, 150, 50)
    p = st.sidebar.number_input("Phosphorus (P)", 0, 150, 50)
    k = st.sidebar.number_input("Potassium (K)", 0, 150, 50)
    temp = st.sidebar.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.sidebar.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    return pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], columns=FEATURES)

input_df = user_inputs()

# Sidebar option to show training metrics when artifacts are loaded
show_metrics = st.sidebar.checkbox("Show training metrics", value=False)

st.write("### Your input", input_df)

# --- Model training / persistence ---
MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"
LE_FILE = "le.joblib"

def train_and_persist(df):
    X = df[FEATURES].values
    y = df['label'].values

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # metrics
    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # persist
    try:
        joblib.dump(clf, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(le, LE_FILE)
    except Exception:
        # if persisting fails, continue without crashing (use in-memory artifacts)
        pass

    return {
        'model': clf,
        'scaler': scaler,
        'le': le,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm
    }


def load_artifacts_or_train(df):
    # if artifact files exist, load them
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(LE_FILE):
        try:
            clf = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            le = joblib.load(LE_FILE)

            # no metrics available when loading only; set placeholders
            return {
                'model': clf,
                'scaler': scaler,
                'le': le,
                'accuracy': None,
                'report': None,
                'confusion_matrix': None
            }
        except Exception:
            # fall back to training if load fails
            return train_and_persist(df)
    else:
        return train_and_persist(df)

with st.spinner('Preparing model (this may train once on first run) ...'):
    artifacts = load_artifacts_or_train(data)

metrics_file = "metrics.json"
# Decide whether to show the Model performance section
should_show_metrics = False
if artifacts.get('accuracy') is not None:
    should_show_metrics = True
elif show_metrics and os.path.exists(metrics_file):
    should_show_metrics = True

if should_show_metrics:
    st.subheader("Model performance")
    if artifacts.get('accuracy') is not None:
        st.write(f"Accuracy (test): **{artifacts['accuracy']:.3f}**")
        if st.expander("Detailed classification report"):
            st.text(artifacts['report'])
    else:
        # show saved metrics from metrics.json
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.info("Model artifacts were loaded from disk — showing last saved training metrics below.")
            if metrics.get("accuracy") is not None:
                st.write(f"Accuracy (test): **{metrics['accuracy']:.3f}**  ")
            if metrics.get("report"):
                if st.expander("Detailed classification report (saved)"):
                    st.text(metrics['report'])
            if metrics.get("trained_at"):
                st.caption(f"Last trained: {metrics['trained_at']}")
        except Exception:
            st.info("Model artifacts were loaded from disk — no training metrics available in this run.")

# --- Make prediction for user input ---
scaled_input = artifacts['scaler'].transform(input_df.values)
pred_idx = artifacts['model'].predict(scaled_input)[0]
pred_label = artifacts['le'].inverse_transform([pred_idx])[0]

st.subheader("Recommended crop")
st.success(f"🌻 You should consider planting: **{pred_label}**")

st.caption("Built with ❤️ using Streamlit & Scikit-learn")
