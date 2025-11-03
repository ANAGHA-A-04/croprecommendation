# AI Crop Recommendation System

This is a small Streamlit app that recommends a crop to grow based on soil and climate inputs (N, P, K, temperature, humidity, pH, rainfall) using a Random Forest classifier trained on the Crop Recommendation dataset.

## Files
- `app.py` - The Streamlit app. Trains a RandomForest model at startup and predicts the best crop from user inputs.
- `Crop_recommendation.csv` - Dataset (the repo includes it locally; if missing the app fetches a fallback URL).
- `requirements.txt` - Python dependencies.

## Setup (Windows, cmd.exe) — recommended minimal steps
1. Make sure you have Python 3.11 installed.
2. Open cmd.exe and create a virtual environment (recommended):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

3. Upgrade pip and essential build tools (helps avoid many install errors):

```cmd
python -m pip install --upgrade pip setuptools wheel
```

4. Install requirements:

```cmd
pip install -r requirements.txt
```

Notes about Streamlit install errors on Windows/Python 3.11:
- If you get errors installing `streamlit`, first make sure pip/setuptools/wheel are upgraded (see step 3).
- If a binary wheel isn't available for an optional dependency, pip may try to compile from source and require Microsoft C++ Build Tools. Installing the latest Streamlit (via `pip install --upgrade streamlit`) usually pulls compatible pre-built wheels.
- As an alternative, use a conda environment which can simplify binary dependency installation.

## Run locally
From the project root (with the venv active):

```cmd
streamlit run app.py
```

The app will open in your default browser and allow you to change sidebar inputs and see the recommended crop.

## Deploy to Streamlit Cloud (fast)
1. Create a GitHub repo and push the project (include `app.py`, `requirements.txt`, and `Crop_recommendation.csv`).
2. Sign into https://streamlit.io/cloud and link your GitHub repo.
3. Pick the branch and the `app.py` file; Streamlit Cloud will install the dependencies and run the app.

If Streamlit Cloud fails to install packages, check the install logs there — often the fix is the same: upgrade pip or switch to a different Python runtime (Streamlit Cloud lets you change the Python version in Advanced settings).

## Troubleshooting
- If `pip install` fails with compilation errors, try:

```cmd
pip install --upgrade pip setuptools wheel
pip install --upgrade streamlit
```

- If problems persist, create a fresh virtual environment and retry, or use conda.

## Next steps / Improvements
- Persist the trained model to disk (`joblib.dump`) so the app doesn't retrain every cold-start.
- Add explainability (SHAP) to show why a crop was recommended.
- Add unit tests for prediction pipeline.

If you want, I can:
- Push these files to a new GitHub repo for you (you'll have to provide credentials or do the final push),
- Walk through deployment to Streamlit Cloud step-by-step,
- Or convert the app to use a saved model so cold-start is faster.
