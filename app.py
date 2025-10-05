# app.py - Gradio app for Car Insurance Claim Prediction (CatBoost with Risk Score)

import gradio as gr
import pandas as pd
import joblib

# ---------------------------
# Load CatBoost model and metrics
# ---------------------------
cat_model = joblib.load("saved_models/catboost_weighted_tuned.pkl")
cat_metrics = joblib.load("saved_metrics/catboost_weighted_tuned_metrics.pkl")

# Ensure metric keys are consistent
cat_metrics = {k.lower().replace(" ", "_"): v for k, v in cat_metrics.items()}

# Leaderboard (CatBoost only)
leaderboard_metrics = {
    "Test Accuracy": cat_metrics.get("test_accuracy", 0),
    "F1 (Test)": cat_metrics.get("f1_(test)", 0),
    "ROC-AUC (Test)": cat_metrics.get("roc-auc_(test)", 0),
    "CV Mean F1": cat_metrics.get("cv_mean_f1", 0),
}

# Columns model was trained on
train_columns = cat_metrics.get("train_columns", [])

# ---------------------------
# Prediction Function
# ---------------------------
def predict_claim(age, gender, vehicle_type, driving_experience, education, income,
                  credit_score, vehicle_year, married, children,
                  annual_mileage, risk_score, vehicle_ownership):
    """
    Prepare input data in the same way preprocessing did, then predict with CatBoost.
    """

    # --- Match preprocessing steps ---
    # Binary columns → "Yes"/"No" (EXCEPT vehicle_ownership which stays numeric)
    married = "Yes" if married == 1 else "No"
    children = "Yes" if children == 1 else "No"
    vehicle_ownership = int(vehicle_ownership)  # keep numeric (0 = Leased, 1 = Owned)

    # Vehicle year → keep string labels ("before 2015"/"after 2015") because CatBoost handles categories
    vehicle_year = str(vehicle_year).strip().lower()

    # Categorical cleanup (strip, lowercase)
    age = str(age).strip().lower()
    gender = str(gender).strip().lower()
    vehicle_type = str(vehicle_type).strip().lower()
    driving_experience = str(driving_experience).strip().lower()
    education = str(education).strip().lower()
    income = str(income).strip().lower()

    # Build input DataFrame
    input_dict = {
        "age": age,
        "gender": gender,
        "vehicle_type": vehicle_type,
        "driving_experience": driving_experience,
        "education": education,
        "income": income,
        "credit_score": credit_score,
        "vehicle_year": vehicle_year,
        "married": married,
        "children": children,
        "annual_mileage": annual_mileage,
        "risk_score": risk_score,
        "vehicle_ownership": vehicle_ownership,
    }

    df_input = pd.DataFrame([input_dict])

    # Reindex to match training columns
    df_input = df_input.reindex(columns=train_columns, fill_value=0)

    # Predict with CatBoost
    prediction = cat_model.predict(df_input)[0]
    prob = cat_model.predict_proba(df_input)[0][1] * 100

    outcome = "✅ Claim Likely" if prediction == 1 else "❌ No Claim"
    metrics_str = "\n".join([f"{k}: {v:.2f}" for k, v in leaderboard_metrics.items()])

    return f"Prediction: {outcome}\nConfidence: {prob:.2f}%\n\nModel Performance:\n{metrics_str}"


# ---------------------------
# Gradio Interface
# ---------------------------
with gr.Blocks(title="Car Insurance Claim Predictor") as demo:
    gr.Markdown("# 🚗 Car Insurance Claim Prediction (CatBoost)")
    gr.Markdown("This app mirrors the exact preprocessing pipeline (binary encoding, categorical standardization, risk score feature).")

    gr.Markdown("### 📊 Model Performance Metrics")
    for k, v in leaderboard_metrics.items():
        gr.Markdown(f"- {k}: {v:.2f}")

    with gr.Row():
        with gr.Column():
            age = gr.Dropdown(['16-25', '26-39', '40-64', '65+'], label="Age")
            gender = gr.Dropdown(['male', 'female'], label="Gender")
            driving_experience = gr.Dropdown(['0-9y', '10-19y', '20-29y', '30y+'], label="Driving Experience")
            education = gr.Dropdown(['none', 'high school', 'university'], label="Education")
            income = gr.Dropdown(['poverty', 'working class', 'middle class', 'upper class'], label="Income")
            vehicle_year = gr.Dropdown(['before 2015', 'after 2015'], label="Vehicle Year")
            vehicle_type = gr.Dropdown(['sedan', 'sports car'], label="Vehicle Type")

        with gr.Column():
            credit_score = gr.Number(label="Credit Score (0-1)", value=0.5, minimum=0, maximum=1)
            vehicle_ownership = gr.Dropdown([0, 1], label="Vehicle Ownership (0 = Leased, 1 = Owned)", value=1)
            annual_mileage = gr.Number(label="Annual Mileage", value=12000)
            risk_score = gr.Number(label="Risk Score", value=0, minimum=0)
            married = gr.Dropdown([0, 1], label="Married (0 = No, 1 = Yes)")
            children = gr.Dropdown([0, 1], label="Children (0 = No, 1 = Yes)")

    predict_btn = gr.Button("Predict Claim")
    output = gr.Textbox(label="Prediction Result", lines=10)

    predict_btn.click(
        predict_claim,
        inputs=[age, gender, vehicle_type, driving_experience, education, income,
                credit_score, vehicle_year, married, children,
                annual_mileage, risk_score, vehicle_ownership],
        outputs=output
    )

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
    