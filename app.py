# app.py (UPDATED) — ASD Screening Support Tool (Prototype)
# Fixes: ValueError when model predicts string labels like "Not ASD"/"ASD"

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ASD Screening Support Tool", layout="centered")
st.title("ASD Screening Support Tool (Prototype)")
st.caption("This is a screening support prototype. Not a clinical diagnosis tool.")

# Load trained bundle (created by train_and_save.py)
bundle = joblib.load("asd_screening_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
rfe = bundle["rfe"]
encoders = bundle["encoders"]
raw_cols = bundle["raw_columns"]
cat_cols = bundle["cat_cols"]

st.subheader("Enter screening information")

# Defaults for numeric inputs (adjust if needed)
numeric_defaults = {
    "age": 5,
    "screening_tool_score": 6,
    "eye_contact_score": 5,
    "speech_delay_score": 5,
    "repetitive_behavior_score": 5,
    "sensory_sensitivity_score": 5,
    "social_interaction_score": 5,
    "red_flag_score": 3,
    "mchat_score": 3,
    "sleep_issue_score": 2,
    "anxiety_score": 2,
}

inputs = {}

# Create input widgets in the same order as training raw columns
for col in raw_cols:
    if col in cat_cols:
        # Use the encoder's known categories
        classes = list(encoders[col].classes_)
        inputs[col] = st.selectbox(col, classes)
    else:
        default = float(numeric_defaults.get(col, 0))
        inputs[col] = st.number_input(col, value=default)

if st.button("Predict ASD Risk"):
    # Build 1-row dataframe
    row = pd.DataFrame([inputs], columns=raw_cols)

    # Encode categorical using stored encoders
    for col in cat_cols:
        le = encoders[col]
        val = row.loc[0, col]
        if val not in le.classes_:
            st.error(f"Invalid category for {col}: {val}")
            st.stop()
        row[col] = le.transform([val])[0]

    # Apply RFE and scaling
    X = rfe.transform(row)
    X = scaler.transform(X)

    # Predict
    pred_raw = model.predict(X)[0]

    # Convert prediction to a readable label
    if isinstance(pred_raw, str):
        pred_label = pred_raw  # e.g., "ASD" or "Not ASD"
    else:
        pred_label = "ASD" if int(pred_raw) == 1 else "Not ASD"

    # Get ASD probability safely (works for string or numeric classes)
    classes = list(model.classes_)  # could be ["Not ASD","ASD"] or [0,1]
    if "ASD" in classes:
        asd_index = classes.index("ASD")
    elif 1 in classes:
        asd_index = classes.index(1)
    else:
        # fallback: assume positive class is last
        asd_index = -1

    proba_asd = float(model.predict_proba(X)[0][asd_index])

    # Display result
    st.subheader("Result")
    if pred_label.strip().lower() == "asd":
        st.error(f"Predicted: ASD risk\n\nConfidence (ASD): {proba_asd:.2%}")
    else:
        st.success(f"Predicted: Not ASD\n\nConfidence (ASD): {proba_asd:.2%}")

    st.info("Reminder: This output is screening support only. Clinical diagnosis requires professionals.")