Car Insurance Claim Prediction Project

Overview

This project implements a machine learning pipeline to predict car insurance claims based on the Car_Insurance_Claim.csv dataset. The pipeline includes data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model training with multiple algorithms (Logistic Regression, Random Forest, CatBoost), evaluation, and deployment via a Gradio web interface. The goal is to predict the binary OUTCOME (0: no claim, 1: claim) with high F1-score and ROC-AUC, addressing class imbalance and ensuring reproducibility.


LATEST UPDATE:

Improvements and Changes from the First Version

The first version focused on basic data loading, EDA, preprocessing, and model experiments. The second version enhances this with better organization, error handling, and user-facing features. Key changes:


Logging System: Added a logging setup with file and console handlers for better debugging and traceability. Logs are saved to pipeline.log.

Modular Structure: Refactored the code into clearly defined blocks with storytelling comments. Introduced functions like setup_logger, evaluate_model, save_best_result, and run_model_variants to reduce repetition and improve readability.

Data Handling: Improved data loading with explicit type conversions and binary encoding for consistency (e.g., married and children as "Yes"/"No").

Feature Engineering: Retained risk_score creation but ensured consistent handling across models.

Model Experiments: Unified experiment logic for all models using run_model_variants, adding CV metrics for all variants.

Feature Importance: Added computation and visualization for the best-tuned models (saved in plots/feature_importance).


Leaderboard: Created a leaderboard from the best models, saved as results/best_models.json, with bar chart visualization in plots/leaderboard_comparison_dynamic.png.



Gradio App: Added an interactive web interface for predictions using the best CatBoost model, with inputs matching the preprocessing pipeline.



Results Saving: Expanded saving of results, including all experiments (results/{model_type}_results.csv) and best models (results/{model_type}_best_model.csv).


Plot Directories: New directories like plots/feature_importance for importance plots, and plots/{model_type} for model-specific visualizations.

The structure changed from a linear script to a more modular one, with functions handling repeated tasks (e.g., tuning, evaluation), reducing code duplication and improving maintainability.


Project Structure

car-insurance-claim-prediction/
│
├── Car_Insurance_Claim.csv          # Raw dataset
├── saved_data/                      # Cleaned data (PKL files)
├── saved_models/                    # Trained models (PKL files)
├── saved_metrics/                   # Metrics for best model (PKL files)
├── plots/                           # General plots
├── plots/eda/                       # Initial EDA visualizations
├── plots/after_cleaning/            # Post-preprocessing visualizations
├── plots/randomforest/              # Random Forest evaluation plots
├── plots/logreg/                    # Logistic Regression evaluation plots
├── plots/catboost/                  # CatBoost evaluation plots
├── plots/feature_importance/        # Feature importance plots (new in v2)
├── results/                         # Experiment results (CSV, JSON) (new in v2)
├── pipeline.log                     # Script logs (new in v2)
├── car_insurance_pipeline.py        # Main script (improved v2)
└── README.md                        # This file



FIRST UPDATE:

Refactor: Separate Gradio app from training script, update prediction input and metrics

Separated Gradio app from main training script; prediction UI and logic now in app.py
Added vehicle_ownership to Gradio UI and prediction function
Removed vehicle_type from UI and prediction
Mapped vehicle_year to numeric in prediction to match training
Lowercased/stripped categorical inputs for consistency
Fixed leaderboard metrics key matching in app
Save and display CV Mean F1 in metrics
Removed share=True from Gradio launch (runs locally)
Updated README with new usage instructions for app.py


Usage

Run the Pipeline:

python car_insurance_pipeline.py

This performs:
Data loading and EDA (saves plots to plots/eda).
Preprocessing and feature engineering (saves data to saved_data).
Model training/evaluation (saves models to saved_models, results to results).
Generates visualizations (saved to plots/{model} and plots/feature_importance).



Creates leaderboard (saved to plots/leaderboard_comparison_dynamic.png).



Launches Gradio app (URL provided in console).



Gradio Interface:

After running, access the web interface via the provided URL.
Input features (e.g., age, credit_score) to predict claim likelihood.
iew prediction, confidence, and model metrics.

Load Saved Models:

import joblib
model = joblib.load('saved_models/catboost_weighted_tuned.pkl')

Results
Best Model: CatBoost (Weighted + Tuned).

Performance (on test set, approximate):

F1 Score: ~0.72-0.75
ROC-AUC: ~0.87-0.88


Metrics saved in saved_metrics/catboost_weighted_tuned_metrics.pkl.


Leaderboard: Compares best models (plots/leaderboard_comparison_dynamic.png).


Feature Importance: Saved in plots/feature_importance for each model.


Imbalance Handling: SMOTE for Random Forest, class weights for Logistic Regression/CatBoost.

Visualizations: Confusion matrices, ROC curves, feature distributions (in plots/).


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
├── car-insurance-claim-prediction.py.py                    # Main script
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
Implement SHAP for feature importance.



Monitor data drift in production.


Contributing

Fork the repository.

Create a feature branch 

Commit changes 

Push to the branch 

Open a pull request.


License

MIT License. See LICENSE file for details.

Acknowledgments
Built with Python, Scikit-learn, CatBoost, and Gradio.

Dataset: Car_Insurance_Claim.csv (source not specified; assumed public).
