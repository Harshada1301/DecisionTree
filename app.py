"""
LoanLens - Loan approval prediction web app (Flask + scikit-learn DecisionTree).

Run locally :  python app.py
Production  :  gunicorn app:app   (used on Render)
"""
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loanlens")

app = Flask(__name__)


def find_model_file() -> Path:
    """Locate the .pkl no matter where it was placed in the repo.

    Checks (case-insensitively) the repo root and ./model first, then
    searches the whole project for any file ending in .pkl whose name
    contains 'decisiontree'; finally falls back to any single .pkl.
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    all_pkls = [p for p in BASE_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pkl"]
    log.info("Files in project root: %s", sorted(p.name for p in BASE_DIR.iterdir()))
    log.info("Found .pkl files: %s", [str(p.relative_to(BASE_DIR)) for p in all_pkls])

    for p in all_pkls:
        if "decisiontree" in p.name.lower().replace("_", "").replace("-", "").replace(" ", ""):
            return p
    if all_pkls:
        return all_pkls[0]
    raise FileNotFoundError(
        "No .pkl model file found in the repository. Commit your model file "
        "(e.g. model/decisiontree.pkl) to GitHub and redeploy."
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_PATH = find_model_file()
log.info("Loading model from %s", MODEL_PATH)
with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)

FEATURE_ORDER = list(model.feature_names_in_)
log.info("Model loaded. Features: %s", FEATURE_ORDER)

# --------------------------------------------------------------------------- #
# Categorical encodings
# The model was trained on label-encoded columns (alphabetical order, the
# default behaviour of sklearn's LabelEncoder). If your notebook used a
# different mapping, edit the dictionaries below.
# --------------------------------------------------------------------------- #
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME_OWNERSHIP = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
LOAN_INTENT = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5,
}
PREV_DEFAULT = {"No": 0, "Yes": 1}

# Human-friendly labels for the UI (value -> label)
UI_OPTIONS = {
    "genders": [("female", "Female"), ("male", "Male")],
    "educations": [(k, k) for k in EDUCATION],
    "homes": [("RENT", "Rent"), ("OWN", "Own"), ("MORTGAGE", "Mortgage"), ("OTHER", "Other")],
    "intents": [
        ("PERSONAL", "Personal"),
        ("EDUCATION", "Education"),
        ("MEDICAL", "Medical"),
        ("VENTURE", "Business venture"),
        ("HOMEIMPROVEMENT", "Home improvement"),
        ("DEBTCONSOLIDATION", "Debt consolidation"),
    ],
}

# (min, max) numeric validation
NUMERIC_RULES = {
    "person_age": (18, 100),
    "person_income": (1, 10_000_000),
    "person_emp_exp": (0, 80),
    "loan_amnt": (1, 5_000_000),
    "loan_int_rate": (0.1, 40),
    "cb_person_cred_hist_length": (0, 80),
    "credit_score": (300, 900),
}


class ValidationError(ValueError):
    pass


def _number(data: dict, key: str) -> float:
    raw = data.get(key)
    if raw is None or str(raw).strip() == "":
        raise ValidationError(f"'{key}' is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"'{key}' must be a number.")
    lo, hi = NUMERIC_RULES[key]
    if not lo <= value <= hi:
        raise ValidationError(f"'{key}' must be between {lo} and {hi}.")
    return value


def _choice(data: dict, key: str, mapping: dict) -> int:
    raw = data.get(key)
    if raw not in mapping:
        raise ValidationError(f"'{key}' has an invalid value.")
    return mapping[raw]


def build_features(data: dict) -> pd.DataFrame:
    income = _number(data, "person_income")
    amount = _number(data, "loan_amnt")
    if _number(data, "person_emp_exp") > _number(data, "person_age") - 14:
        raise ValidationError("Work experience is not realistic for this age.")

    row = {
        "person_age": _number(data, "person_age"),
        "person_gender": _choice(data, "person_gender", GENDER),
        "person_education": _choice(data, "person_education", EDUCATION),
        "person_income": income,
        "person_emp_exp": _number(data, "person_emp_exp"),
        "person_home_ownership": _choice(data, "person_home_ownership", HOME_OWNERSHIP),
        "loan_amnt": amount,
        "loan_intent": _choice(data, "loan_intent", LOAN_INTENT),
        "loan_int_rate": _number(data, "loan_int_rate"),
        # derived exactly like the training data (rounded to 2 decimals)
        "loan_percent_income": round(amount / income, 2),
        "cb_person_cred_hist_length": _number(data, "cb_person_cred_hist_length"),
        "credit_score": _number(data, "credit_score"),
        "previous_loan_defaults_on_file": _choice(data, "previous_loan_defaults_on_file", PREV_DEFAULT),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def risk_notes(row: pd.Series) -> list:
    """Plain-language factors shown under the result."""
    notes = []
    score = row["credit_score"]
    if score >= 740:
        notes.append({"tone": "good", "text": "Excellent credit score"})
    elif score >= 670:
        notes.append({"tone": "ok", "text": "Good credit score"})
    else:
        notes.append({"tone": "bad", "text": "Credit score below 670"})

    lpi = row["loan_percent_income"]
    if lpi <= 0.20:
        notes.append({"tone": "good", "text": f"Loan is {lpi:.0%} of income - comfortable"})
    elif lpi <= 0.35:
        notes.append({"tone": "ok", "text": f"Loan is {lpi:.0%} of income - moderate"})
    else:
        notes.append({"tone": "bad", "text": f"Loan is {lpi:.0%} of income - high burden"})

    if row["previous_loan_defaults_on_file"] == 1:
        notes.append({"tone": "bad", "text": "Previous default on file"})
    else:
        notes.append({"tone": "good", "text": "No previous defaults"})

    if row["loan_int_rate"] >= 15:
        notes.append({"tone": "bad", "text": f"High interest rate ({row['loan_int_rate']:.1f}%)"})
    return notes


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html", **UI_OPTIONS)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        features = build_features(data)
    except ValidationError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    try:
        proba = model.predict_proba(features)[0]
        pred = int(model.predict(features)[0])
    except Exception:  # pragma: no cover
        log.exception("Prediction failed")
        return jsonify(ok=False, error="Prediction failed. Please try again."), 500

    classes = list(model.classes_)
    approve_p = float(proba[classes.index(1)])
    return jsonify(
        ok=True,
        approved=bool(pred == 1),
        approval_probability=round(approve_p * 100, 1),
        confidence=round(float(max(proba)) * 100, 1),
        loan_percent_income=float(features.loc[0, "loan_percent_income"]),
        notes=risk_notes(features.iloc[0]),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
