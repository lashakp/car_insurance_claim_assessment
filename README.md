Car Insurance Claim Prediction Project

Overview

This project implements a machine learning pipeline to predict car insurance claims based on the Car_Insurance_Claim.csv dataset. The pipeline includes data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model training with multiple algorithms (Logistic Regression, Random Forest, CatBoost), evaluation, and deployment via a Gradio web interface. The goal is to predict the binary OUTCOME (0: no claim, 1: claim) with high F1-score and ROC-AUC, addressing class imbalance and ensuring reproducibility.


Refactor: Separate Gradio app from training script, update prediction input and metrics

- Separated Gradio app from main training script; prediction UI and logic now in app.py
- Added vehicle_ownership to Gradio UI and prediction function
- Removed vehicle_type from UI and prediction
- Mapped vehicle_year to numeric in prediction to match training
- Lowercased/stripped categorical inputs for consistency
- Fixed leaderboard metrics key matching in app
- Save and display CV Mean F1 in metrics
- Removed share=True from Gradio launch (runs locally)
- Updated README with new usage instructions for app.py

Features





Comprehensive EDA: Visualizations (histograms, boxplots, countplots, heatmaps) to analyze feature distributions, target imbalance, and correlations.



Data Preprocessing: Cleaning column names, standardizing categoricals, dropping irrelevant columns, clipping outliers, imputing missing values.



Feature Engineering: Created risk_score by summing driving violations/accidents.



Model Experiments:





Logistic Regression (Baseline, Weighted, Tuned, Weighted + Tuned).



Random Forest (Baseline, SMOTE, Tuned, SMOTE + Tuned).



CatBoost (Baseline, Weighted, Tuned, Weighted + Tuned; top performer).



Evaluation: Metrics include Accuracy, Precision, Recall, F1, Specificity, ROC-AUC; visualized via confusion matrices and ROC curves.



Deployment: Gradio interface for user-friendly predictions.



Persistence: Models, results, and cleaned data saved as PKL/CSV files.

Dataset





File: Car_Insurance_Claim.csv



Rows/Columns: 10,000 rows, 19 columns (mix of numeric and categorical).



Target: OUTCOME (imbalanced: 68.67% no claims, 31.33% claims).



Key Features: credit_score, annual_mileage, age, gender, driving_experience, etc.



Issues: Missing values in CREDIT_SCORE (982) and ANNUAL_MILEAGE (957); handled via imputation.

Project Structure

car-insurance-claim-prediction/
│
├── Car_Insurance_Claim.csv          # Raw dataset
├── saved_data/                      # Cleaned data (PKL files)
├── saved_models/                    # Trained models (PKL files)
├── saved_results/                   # Model results (CSV files)
├── saved_metrics/                   # Metrics for best model (PKL files)
├── plots/                           # General plots
├── plots/eda/                       # Initial EDA visualizations
├── plots/after_cleaning/            # Post-preprocessing visualizations
├── plots/randomforest/              # Random Forest evaluation plots
├── plots/logreg/                    # Logistic Regression evaluation plots
├── plots/catboost/                  # CatBoost evaluation plots
├── untitled31.py                    # Main script
└── README.md                        # This file

Installation





Clone the repository:

git clone https://github.com/lashakp/car_insurance_claim_assessment.
cd paul-akporarhe-assessment



Install dependencies:

pip install -r requirements.txt

Or manually install:

pip install pandas numpy matplotlib seaborn scikit-learn catboost imblearn joblib gradio



Ensure Car_Insurance_Claim.csv is in the root directory.

Usage





Run the Pipeline:

python car-insurance-claim-prediction.py

This executes:





Data loading and EDA (saves plots to plots/eda).



Preprocessing and feature engineering (saves data to saved_data).



Model training/evaluation (saves models to saved_models, results to saved_results).



Generates visualizations (saved to plots/[model]).



Launches Gradio app (URL provided in console).



Gradio Interface:





After running, access the web interface via the provided URL.



Input features (e.g., age, credit_score) to predict claim likelihood.



View prediction, confidence, and model metrics.



Load Saved Models:

import joblib
model = joblib.load('saved_models/catboost_weighted_tuned.pkl')

Results





Best Model: CatBoost (Weighted + Tuned).



Performance (on test set, approximate):





F1 Score: ~0.75-0.80



ROC-AUC: ~0.85-0.90



Metrics saved in saved_metrics/catboost_weighted_tuned_metrics.pkl.



Leaderboard: Compares best models (plots/leaderboard_comparison_dynamic.png).



Imbalance Handling: SMOTE for Random Forest, class weights for Logistic Regression/CatBoost.



Visualizations: Confusion matrices, ROC curves, feature distributions (in plots/).

Notes





Reproducibility: Fixed random_state=42 ensures consistent results.



Dependencies: Tested with Python 3.8+. Ensure catboost is installed for CatBoost models.



Troubleshooting:





Missing file errors: Verify Car_Insurance_Claim.csv is in the root.



Gradio issues: Check internet connection for share=True or run locally (share=False).



Memory errors: Reduce n_iter in RandomizedSearchCV or use fewer CV folds.



Future Improvements:





Add XGBoost experiments (imported but unused).



Implement SHAP for feature importance.



Monitor data drift in production.

Contributing






Open a pull request.

License

MIT License. See LICENSE file for details.

Acknowledgments





Built with Python, Scikit-learn, CatBoost, and Gradio.



Dataset: Car_Insurance_Claim.csv (source not specified; assumed public).
