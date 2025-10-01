import gradio as gr
import pandas as pd
import joblib
import os

# ---------------------------
# Load saved CatBoost model and metrics
# ---------------------------
cat_model = joblib.load('saved_models/catboost_weighted_tuned.pkl')
cat_metrics = joblib.load('saved_metrics/catboost_weighted_tuned_metrics.pkl')

# Normalize metric keys to avoid KeyErrors
cat_metrics = {k.lower().replace(" ", "_"): v for k, v in cat_metrics.items()}

# Select leaderboard metrics safely
leaderboard_metrics = {
    "Test Accuracy": cat_metrics.get("test_accuracy", 0),
    "F1 (Test)": cat_metrics.get("f1_(test)", 0),
    "ROC-AUC (Test)": cat_metrics.get("roc-auc_(test)", 0),
    "CV Mean F1": cat_metrics.get("cv_mean_f1", 0)
}

def predict_claim(age, gender, driving_experience, education, income,
                  credit_score, vehicle_year, married, children,
                  annual_mileage, risk_score, vehicle_ownership):
    # Lowercase and strip categorical values to match training
    age = str(age).strip().lower()
    gender = str(gender).strip().lower()
    driving_experience = str(driving_experience).strip().lower()
    education = str(education).strip().lower()
    income = str(income).strip().lower()
    # Map vehicle_year to numeric as in training
    vehicle_year_map = {'before 2015': 0, 'after 2015': 1}
    vehicle_year = vehicle_year_map.get(str(vehicle_year).strip().lower(), 0)

    # Build input DataFrame with only the columns used in training
    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'vehicle_ownership': [vehicle_ownership],  # <-- add this
        'married': [married],
        'children': [children],
        'annual_mileage': [annual_mileage],
        'risk_score': [risk_score],
        'age': [age],
        'gender': [gender],
        'driving_experience': [driving_experience],
        'education': [education],
        'income': [income],
        'vehicle_year': [vehicle_year]
    })

    train_columns = cat_metrics.get("train_columns", [])
    input_data = input_data[[col for col in train_columns if col in input_data.columns]]

    prediction = cat_model.predict(input_data)[0]
    probability = cat_model.predict_proba(input_data)[0][1] * 100

    outcome = "Claim Likely (1)" if prediction == 1 else "No Claim (0)"
    metrics_str = "\n".join([f"{k}: {v:.2f}" for k, v in leaderboard_metrics.items()])

    return f"Prediction: {outcome}\nConfidence: {probability:.2f}%\n\nModel Performance:\n{metrics_str}"

with gr.Blocks(title="Car Insurance Claim Predictor") as demo:
    gr.Markdown("# Car Insurance Claim Prediction App")
    gr.Markdown("Enter driver and vehicle details to predict if a claim is likely using CatBoost Weighted + Tuned model.")

    gr.Markdown("### Model Performance Metrics (Test Set & CV)")
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
            # vehicle_type removed

        with gr.Column():
            credit_score = gr.Number(label="Credit Score (0-1)", value=0.5, minimum=0, maximum=1)
            vehicle_ownership = gr.Dropdown([0, 1], label="Vehicle Ownership")  # <-- add this
            annual_mileage = gr.Number(label="Annual Mileage", value=12000)
            risk_score = gr.Number(label="Risk Score", value=0, minimum=0)
            married = gr.Dropdown([0, 1], label="Married")
            children = gr.Dropdown([0, 1], label="Children")

    predict_btn = gr.Button("Predict Claim")
    output = gr.Textbox(label="Prediction Result", lines=10)

    predict_btn.click(
        predict_claim,
        inputs=[age, gender, driving_experience, education, income,
                credit_score, vehicle_year, married, children,
                annual_mileage, risk_score, vehicle_ownership],  # <-- add vehicle_ownership here
        outputs=output
    )

demo.launch()  # Remove share=True to run only on your computer