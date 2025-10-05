# --------------------------------
# Block: Import Libraries
# --------------------------------

# ---------------- Core Libraries ----------------
import pandas as pd  # For working with tabular data
import numpy as np  # For numeric operations and arrays

# ---------------- Visualization ----------------
import seaborn as sns  # For statistical plots
import matplotlib.pyplot as plt  # For general plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (important for saving plots in scripts)

# ---------------- Scikit-learn Tools ----------------
from sklearn.model_selection import train_test_split, cross_val_score  # Train/test split + CV
from sklearn.impute import SimpleImputer  # Fill missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encode categorical, scale numeric
from sklearn.compose import ColumnTransformer  # Apply transforms by column type
from sklearn.pipeline import Pipeline  # Chain preprocessing + model
import joblib  # Save datasets, models, encoders

# ---------------- Models ----------------
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from catboost import CatBoostClassifier  # CatBoost (handles categoricals internally)

# ---------------- Evaluation ----------------
from sklearn.metrics import (
    classification_report,  # Precision/recall/F1 summary
    roc_auc_score,  # Area under ROC curve
    confusion_matrix,  # Confusion matrix
    roc_curve,  # ROC curve points
    auc,  # AUC metric
    f1_score,  # F1 score
    RocCurveDisplay,  # For plotting ROC curve
)

# ---------------- Imbalanced Data Handling ----------------
from imblearn.over_sampling import SMOTE  # Handle class imbalance via oversampling

# ---------------- Hyperparameter Search ----------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # Model tuning

import logging

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),  # save logs to file
        logging.StreamHandler()               # print logs to console
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------
# Block: Define File Paths and Constants
# --------------------------------
RAW_PATH = "Car_Insurance_Claim.csv"  # Raw dataset path
CLEANED_PATH = "Car_Insurance_Claim_cleaned.csv"  # Optional: save cleaned dataset
RANDOM_STATE = 42  # Fixed seed for reproducibility
TARGET_COL = "OUTCOME"  # Target column (binary classification)


# --------------------------------
# Block: 1. Load & Backup Raw Data
# --------------------------------
df_raw = pd.read_csv(RAW_PATH)  # Load raw CSV into DataFrame
df = df_raw.copy()  # Copy to preserve original
logging.info(f"Raw data loaded with shape: {df.shape}")


# --------------------------------
# 1. Define target and features, split dataset
# --------------------------------
print("1️⃣ Defining target and features, performing train/test split...")

TARGET_COL = "OUTCOME"  # Target column from dataset
features = [c for c in df_raw.columns if c != TARGET_COL]  # All other columns = features

from sklearn.model_selection import train_test_split

# Stratified split ensures class balance in train/test
X_train, X_test, y_train, y_test = train_test_split(
    df_raw[features], df_raw[TARGET_COL],
    test_size=0.2,  # 80/20 split
    random_state=42,  # Reproducible
    stratify=df_raw[TARGET_COL]  # Maintain class distribution
)

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Keep untouched copy of train data for before/after visualizations
X_train_raw = X_train.copy()

# --------------------------------
# Block: 2. Data Cleaning & Preprocessing

# --------------------------------
# 2. Clean column names
# --------------------------------
print("2️⃣ Cleaning column names...")

def clean_column_names(df):
    # Strip spaces, lowercase, replace spaces with underscores
    return df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)

# Update TARGET_COL if lowercase name exists in df_raw
TARGET_COL = TARGET_COL.lower() if TARGET_COL.lower() in df_raw.columns else TARGET_COL

print("✅ Column names cleaned. Sample columns:", X_train.columns.tolist()[:10])
# --------------------------------

# --------------------------------
# 3. Standardize categorical formats
# --------------------------------
print("3️⃣ Standardizing categorical columns (strip spaces, lowercase)...")

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Make all categorical values lowercase and trim spaces
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).str.strip().str.lower()
    X_test[col] = X_test[col].astype(str).str.strip().str.lower()

print(f"✅ Standardized {len(categorical_cols)} categorical columns: {categorical_cols}")
# --------------------------------

# --------------------------------
# 4. Drop columns not needed
# --------------------------------
print("4️⃣ Dropping unnecessary columns...")

# Columns irrelevant for modeling (IDs, redundant, or high-cardinality)
cols_to_drop = ['id', 'race', 'postal_code']
cols_to_drop_existing = [c for c in cols_to_drop if c in X_train.columns]

X_train = X_train.drop(columns=cols_to_drop_existing)
X_test = X_test.drop(columns=cols_to_drop_existing)

print(f"✅ Dropped columns: {cols_to_drop_existing}")
# --------------------------------

# --------------------------------
# 5. Outlier detection & clipping
# --------------------------------
print("5️⃣ Detecting and clipping outliers in numeric columns...")

numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Apply IQR-based clipping to cap outliers
for col in numeric_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

print(f"✅ Outlier clipping applied to numeric columns: {numeric_cols}")
# --------------------------------

# --------------------------------
# 6. Impute Missing Values
# --------------------------------
print("6️⃣ Imputing missing values...")

from sklearn.impute import SimpleImputer

# Recompute numeric/categorical columns after drops
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Numeric → median imputation
num_imputer = SimpleImputer(strategy='median')
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

# Categorical → most frequent imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

print(f"✅ Missing values imputed for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.")
print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# --------------------------------
# Extra cleaning: Binary consistency
# --------------------------------
if 'married' in X_train.columns:
    X_train['married'] = X_train['married'].map({1: 'Yes', 0: 'No'})
    X_test['married'] = X_test['married'].map({1: 'Yes', 0: 'No'})

if 'children' in X_train.columns:
    X_train['children'] = X_train['children'].map({1: 'Yes', 0: 'No'})
    X_test['children'] = X_test['children'].map({1: 'Yes', 0: 'No'})

# Keep vehicle_ownership numeric (0 = Leased, 1 = Owned)
if 'vehicle_ownership' in X_train.columns:
    X_train['vehicle_ownership'] = X_train['vehicle_ownership'].map({1: 1, 0: 0})
    X_test['vehicle_ownership'] = X_test['vehicle_ownership'].map({1: 1, 0: 0})

print("✅ Cleaned binary columns: married/children as Yes/No, vehicle_ownership as numeric (0/1).")


# --------------------------------
# Feature Engineering
# --------------------------------
print("7️⃣ Performing feature engineering...")
# --------------------------------
# 7. Feature Engineering: Risk Score
# --------------------------------
print("7️⃣ Creating RISK_SCORE from violations/accidents if present...")

risk_components = ['speeding_violations', 'duis', 'past_accidents']
existing_risk_cols = [c for c in risk_components if c in X_train.columns]

if existing_risk_cols:
    # Create risk_score as sum of violations
    X_train['risk_score'] = X_train[existing_risk_cols].sum(axis=1)
    X_test['risk_score'] = X_test[existing_risk_cols].sum(axis=1)

    # Drop original violation columns
    X_train = X_train.drop(columns=existing_risk_cols)
    X_test = X_test.drop(columns=existing_risk_cols)
    print(f"✅ Created RISK_SCORE and dropped columns: {existing_risk_cols}")
else:
    print("⚠️ No risk score components found in the data.")
# --------------------------------

# --------------------------------
# 8. Prepare unencoded CatBoost data
# --------------------------------
print("8️⃣ Preparing unencoded data for CatBoost...")

# CatBoost can handle categorical directly → keep copy before encoding
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

cat_features = ['age', 'gender', 'vehicle_type', 'driving_experience', 'education', 'income', 'vehicle_year']
numeric_cols_cat = [c for c in X_train_cat.columns if c not in cat_features]

print(f"✅ CatBoost categorical features: {cat_features}")
print(f"✅ Numeric features for CatBoost: {numeric_cols_cat}")


# --------------------------------
# 9. Encoding for Logistic Regression / Random Forest
# --------------------------------
from sklearn.preprocessing import OneHotEncoder

# Separate numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# One-hot encode categorical variables
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit encoder on train, transform both train/test
X_train_cat_enc = ohe.fit_transform(X_train[categorical_cols])
X_test_cat_enc = ohe.transform(X_test[categorical_cols])

# Convert encoded arrays to DataFrames
encoded_cols = ohe.get_feature_names_out(categorical_cols)
X_train_cat_enc = pd.DataFrame(X_train_cat_enc, columns=encoded_cols, index=X_train.index)
X_test_cat_enc = pd.DataFrame(X_test_cat_enc, columns=encoded_cols, index=X_test.index)

# Merge numeric + encoded categorical
X_train_encoded = pd.concat([X_train[numeric_cols], X_train_cat_enc], axis=1)
X_test_encoded = pd.concat([X_test[numeric_cols], X_test_cat_enc], axis=1)

print("✅ Encoded feature shape (train):", X_train_encoded.shape)
print("✅ Encoded feature shape (test):", X_test_encoded.shape)
# --------------------------------


# model evaluation functions

# --------------------------------
# Function: evaluate_model
# What: Evaluate classifier on train/test sets, compute full metrics, generate plots.
# Why: Standardize evaluation and visualization across all models for consistency.
# --------------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import seaborn as sns

def evaluate_model(
    model_name,          # String: model name for reporting
    model,               # Classifier instance
    X_train, y_train,    # Training data and labels
    X_test, y_test,      # Test data and labels
    results_list,        # List to append results for comparison
    plot=True,           # Whether to generate confusion matrix + ROC plots
    cat_features=None,   # Reserved for CatBoost if needed
    is_tuned=False,      # Track if this model is from hyperparameter tuning
    cv_search_object=None # Store search object for tuned models (optional)
):
    print(f"\nEvaluating model: {model_name}")
    logger.info(f"Evaluating model: {model_name}")

    # ---------------------------
    # Train model
    # ---------------------------
    model.fit(X_train, y_train)

    # ---------------------------
    # Predictions
    # ---------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # If model supports predict_proba, use probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = y_test_pred  # fallback to hard predictions

    # ---------------------------
    # Compute metrics
    # ---------------------------
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)

    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Specificity = True Negative Rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp)

    # ROC-AUC score (probability-based metric)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    # ---------------------------
    # Cross-validation
    # ---------------------------
    # 5-fold CV on training set using F1 score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Confusion matrix (for plotting)
    cm = confusion_matrix(y_test, y_test_pred)

    # ---------------------------
    # Print + log metrics
    # ---------------------------
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    logger.info(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    print(f"Train Precision: {train_prec:.4f}, Test Precision: {test_prec:.4f}")
    logger.info(f"Train Precision: {train_prec:.4f}, Test Precision: {test_prec:.4f}")

    print(f"Train Recall: {train_rec:.4f}, Test Recall: {test_rec:.4f}")
    logger.info(f"Train Recall: {train_rec:.4f}, Test Recall: {test_rec:.4f}")

    print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    logger.info(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

    print(f"Specificity (Test): {specificity:.4f}, ROC-AUC (Test): {roc_auc:.4f}")
    logger.info(f"Specificity (Test): {specificity:.4f}, ROC-AUC (Test): {roc_auc:.4f}")

    print(f"CV F1 Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"CV F1 Mean: {cv_mean:.4f} ± {cv_std:.4f}")

    # ---------------------------
    # Save results to list
    # ---------------------------
    result_dict = {
        "Model": model_name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Train Precision": train_prec,
        "Test Precision": test_prec,
        "Train Recall": train_rec,
        "Test Recall": test_rec,
        "Train F1": train_f1,
        "F1 (Test)": test_f1,
        "Specificity (Test)": specificity,
        "ROC-AUC (Test)": roc_auc,
        "CV F1 Mean": cv_mean,
        "CV F1 Std": cv_std
    }
    results_list.append(result_dict)

    # ---------------------------
    # Visualization (optional)
    # ---------------------------
    if plot:
        # Confusion Matrix Heatmap
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} - Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_confusion_matrix.png"))
        plt.close()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")  # baseline
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve (Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_roc_curve.png"))
        plt.close()

    return results_list


# --------------------------------
# Function: save_best_result
# What: Save last result of each experiment + best params if tuned
# Why: Keep per-model best results consistent
# --------------------------------
def save_best_result(results_list, model_name, results_dir="results", search_obj=None):
    os.makedirs(results_dir, exist_ok=True)
    best_result = pd.DataFrame([results_list[-1]])

    # Save metrics
    filename = f"{model_name.replace(' ', '_')}_best.csv"
    best_result.to_csv(os.path.join(results_dir, filename), index=False)

    # Save best params if hyperparameter search
    if search_obj is not None:
        params_file = f"{model_name.replace(' ', '_')}_best_params.txt"
        with open(os.path.join(results_dir, params_file), "w") as f:
            f.write(f"Best Params:\n{search_obj.best_params_}\n")
            f.write(f"Best CV Score: {search_obj.best_score_:.4f}\n")

    print(f"✅ Saved best result for {model_name}")



# --------------------------------
# Block: Random Forest Experiments (4 Variants with SMOTE & Tuning)
# --------------------------------
from sklearn.ensemble import RandomForestClassifier       # Random Forest model
from sklearn.model_selection import RandomizedSearchCV    # Randomized hyperparameter search
from imblearn.over_sampling import SMOTE                  # Handle class imbalance
import pandas as pd
import matplotlib.pyplot as plt
import os

rf_results = []                                           # Store results for all RF experiments
PLOT_DIR = "plots/randomforest"                           # Directory to save plots
RESULTS_DIR = "results"                                   # Directory to save results
os.makedirs(PLOT_DIR, exist_ok=True)                      # Create folder for plots
os.makedirs(RESULTS_DIR, exist_ok=True)                   # Create folder for results

# Use the encoded data for Random Forest (RF requires numeric inputs)
X_train_rf = X_train_encoded
X_test_rf = X_test_encoded

# ---------------------------
# 1) Random Forest - Baseline
# ---------------------------
rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_results = evaluate_model(
    "RandomForest - Baseline",
    rf_baseline,
    X_train_rf, y_train,
    X_test_rf, y_test,
    rf_results,
    plot=True
)
pd.DataFrame([rf_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "RandomForest_Baseline_best.csv"), index=False
)

# ---------------------------
# 2) Random Forest - SMOTE (Untuned)
# ---------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_rf, y_train)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_results = evaluate_model(
    "RandomForest - SMOTE (Untuned)",
    rf_smote,
    X_train_smote, y_train_smote,
    X_test_rf, y_test,
    rf_results,
    plot=True
)
pd.DataFrame([rf_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "RandomForest_SMOTE_Untuned_best.csv"), index=False
)

# ---------------------------
# 3) Random Forest - Tuned (No SMOTE)
# ---------------------------
param_dist_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rand_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist_rf,
    n_iter=5,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_rf.fit(X_train_rf, y_train)

print("\nBest RandomForest (Tuned, No SMOTE) params:", rand_search_rf.best_params_)
print("Best CV F1 (RandomForest Tuned):", rand_search_rf.best_score_)

rf_results = evaluate_model(
    "RandomForest - Tuned (No SMOTE)",
    rand_search_rf.best_estimator_,
    X_train_rf, y_train,
    X_test_rf, y_test,
    rf_results,
    plot=True,
    is_tuned=True,
    cv_search_object=rand_search_rf
)
pd.DataFrame([rf_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "RandomForest_Tuned_NoSMOTE_best.csv"), index=False
)

# ---------------------------
# 4) Random Forest - SMOTE + Tuned
# ---------------------------
rand_search_rf_smote = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist_rf,
    n_iter=5,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_rf_smote.fit(X_train_smote, y_train_smote)

print("\nBest RandomForest (SMOTE + Tuned) params:", rand_search_rf_smote.best_params_)
print("Best CV F1 (RandomForest SMOTE Tuned):", rand_search_rf_smote.best_score_)

rf_results = evaluate_model(
    "RandomForest - SMOTE + Tuned",
    rand_search_rf_smote.best_estimator_,
    X_train_smote, y_train_smote,
    X_test_rf, y_test,
    rf_results,
    plot=True,
    is_tuned=True,
    cv_search_object=rand_search_rf_smote
)
pd.DataFrame([rf_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "RandomForest_SMOTE_Tuned_best.csv"), index=False
)

# ---------------------------
# Summary DataFrame + Bar Chart
# ---------------------------
df_rf_results = pd.DataFrame(rf_results)
print("\n===== Random Forest Experiments Summary =====")
print(df_rf_results)

# Save ALL results summary
rf_results_path = os.path.join(RESULTS_DIR, "RandomForest_results.csv")
df_rf_results.to_csv(rf_results_path, index=False)
print(f"✅ RandomForest results saved to {rf_results_path}")

# Bar chart comparing metrics
plot_cols = [
    "Train Accuracy", "Test Accuracy",
    "Train Precision", "Test Precision",
    "Train Recall", "Test Recall",
    "Train F1", "F1 (Test)", "Specificity (Test)", "ROC-AUC (Test)"
]

available_cols = [c for c in df_rf_results.columns if c in plot_cols]
df_plot = df_rf_results.set_index("Model")[available_cols]

plt.figure(figsize=(14,6))
df_plot.plot(kind="bar")
plt.ylim(0,1)
plt.title("RandomForest Variants - Train/Test Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "RandomForest_variants_metrics_comparison.png"))
plt.close()

save_best_result(rf_results, "RandomForest - Baseline", RESULTS_DIR)
save_best_result(rf_results, "RandomForest - SMOTE (Untuned)", RESULTS_DIR)
save_best_result(rf_results, "RandomForest - Tuned (No SMOTE)", RESULTS_DIR, search_obj=rand_search_rf)
save_best_result(rf_results, "RandomForest - SMOTE + Tuned", RESULTS_DIR, search_obj=rand_search_rf_smote)



# --------------------------------
# Block: Logistic Regression Experiments (4 Variants with Scaling)
# --------------------------------
from sklearn.linear_model import LogisticRegression        # Logistic Regression model
from sklearn.model_selection import RandomizedSearchCV      # Randomized hyperparameter search
from sklearn.preprocessing import StandardScaler            # Scaling for LR
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import os

log_results = []                                            # Store results for all LR experiments
PLOT_DIR = "plots/logreg"                                   # Directory to save plots
RESULTS_DIR = "results"                                     # Directory to save results
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use the encoded data for Logistic Regression
X_train_lr = X_train_encoded.copy()
X_test_lr = X_test_encoded.copy()

# Scale numeric features for Logistic Regression (important for gradient-based solvers)
scaler = StandardScaler()
numeric_cols_encoded = X_train_lr.select_dtypes(include=['number']).columns.tolist()
X_train_lr[numeric_cols_encoded] = scaler.fit_transform(X_train_lr[numeric_cols_encoded])
X_test_lr[numeric_cols_encoded] = scaler.transform(X_test_lr[numeric_cols_encoded])


# ---------------------------
# 1) Logistic Regression - Baseline
# ---------------------------
log_baseline = LogisticRegression(max_iter=1000, solver='saga', random_state=RANDOM_STATE)
log_results = evaluate_model(
    "LogReg - Baseline",
    log_baseline,
    X_train_lr, y_train,
    X_test_lr, y_test,
    log_results,
    plot=True
)
pd.DataFrame([log_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "LogReg_Baseline_best.csv"), index=False
)


# ---------------------------
# 2) Logistic Regression - Weighted
# ---------------------------
log_weighted = LogisticRegression(max_iter=1000, solver='saga', class_weight="balanced", random_state=RANDOM_STATE)
log_results = evaluate_model(
    "LogReg - Weighted",
    log_weighted,
    X_train_lr, y_train,
    X_test_lr, y_test,
    log_results,
    plot=True
)
pd.DataFrame([log_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "LogReg_Weighted_best.csv"), index=False
)


# ---------------------------
# 3) Logistic Regression - Tuned (No Weights)
# ---------------------------
param_dist_log = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2", None],   # saga supports l1, l2, elasticnet, none
    "solver": ["saga", "lbfgs"]
}

rand_search_log = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    param_distributions=param_dist_log,
    n_iter=5,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_log.fit(X_train_lr, y_train)

print("\nBest Logistic Regression (Tuned, No Weights) params:", rand_search_log.best_params_)
print("Best CV F1 (LogReg tuned):", rand_search_log.best_score_)

log_results = evaluate_model(
    "LogReg - Tuned (No Weights)",
    rand_search_log.best_estimator_,
    X_train_lr, y_train,
    X_test_lr, y_test,
    log_results,
    plot=True,
    is_tuned=True,
    cv_search_object=rand_search_log
)
pd.DataFrame([log_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "LogReg_Tuned_NoWeights_best.csv"), index=False
)


# ---------------------------
# 4) Logistic Regression - Weighted + Tuned
# ---------------------------
rand_search_log_weighted = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, solver='saga', class_weight="balanced", random_state=RANDOM_STATE),
    param_distributions=param_dist_log,
    n_iter=5,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_log_weighted.fit(X_train_lr, y_train)

print("\nBest Logistic Regression (Weighted + Tuned) params:", rand_search_log_weighted.best_params_)
print("Best CV F1 (LogReg Tuned + Weighted):", rand_search_log_weighted.best_score_)

log_results = evaluate_model(
    "LogReg - Weighted + Tuned",
    rand_search_log_weighted.best_estimator_,
    X_train_lr, y_train,
    X_test_lr, y_test,
    log_results,
    plot=True,
    is_tuned=True,
    cv_search_object=rand_search_log_weighted
)
pd.DataFrame([log_results[-1]]).to_csv(
    os.path.join(RESULTS_DIR, "LogReg_Weighted_Tuned_best.csv"), index=False
)


# ---------------------------
# Summary DataFrame + Bar Chart
# ---------------------------
df_log_results = pd.DataFrame(log_results)
print("\n===== Logistic Regression Experiments Summary =====")
print(df_log_results)

# Save ALL results summary
log_results_path = os.path.join(RESULTS_DIR, "LogReg_results.csv")
df_log_results.to_csv(log_results_path, index=False)
print(f"✅ Logistic Regression results saved to {log_results_path}")

# Bar chart comparing metrics
plot_cols = [
    "Train Accuracy", "Test Accuracy",
    "Train Precision", "Test Precision",
    "Train Recall", "Test Recall",
    "Train F1", "F1 (Test)", "ROC-AUC (Test)"
]

available_cols = [c for c in df_log_results.columns if c in plot_cols]
df_plot = df_log_results.set_index("Model")[available_cols]

plt.figure(figsize=(14,6))
df_plot.plot(kind="bar")
plt.ylim(0,1)
plt.title("Logistic Regression Variants - Train/Test Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "LogReg_variants_metrics_comparison.png"))
plt.close()

save_best_result(log_results, "LogReg - Baseline", RESULTS_DIR)
save_best_result(log_results, "LogReg - Weighted", RESULTS_DIR)
save_best_result(log_results, "LogReg - Tuned (No Weights)", RESULTS_DIR, search_obj=rand_search_log)
save_best_result(log_results, "LogReg - Weighted + Tuned", RESULTS_DIR, search_obj=rand_search_log_weighted)



# =========================================
# Block: CatBoost Experiments (4 Variants)
# =========================================
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Setup
# ---------------------------
cat_results = []  # Store experiment results
PLOT_DIR = "plots/catboost"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------
# Define numeric and categorical features
# ---------------------------
numeric_cols = ['credit_score', 'vehicle_ownership', 'married', 'children', 'annual_mileage', 'risk_score']
cat_features = ['age', 'vehicle_type', 'children', 'married', 'gender', 'driving_experience', 'education', 'income', 'vehicle_year']

# Use the cleaned, unencoded data (CatBoost handles categorical directly)
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

# ---------------------------
# 1) CatBoost Baseline
# ---------------------------
cat_baseline = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    cat_features=cat_features  # CatBoost can natively handle categoricals
)

cat_results = evaluate_model(
    "CatBoost - Baseline",
    cat_baseline,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features
)

# ---------------------------
# 2) CatBoost Weighted
# ---------------------------
cat_weighted = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    class_weights=[1, 2],  # Handle imbalance
    cat_features=cat_features
)

cat_results = evaluate_model(
    "CatBoost - Weighted",
    cat_weighted,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features
)

# ---------------------------
# 3) CatBoost Tuned (No Weights)
# ---------------------------
param_dist_cat = {
    'depth': [4, 6, 8, 10],          # tree depth
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 500, 800],   # boosting rounds
    'l2_leaf_reg': [1, 3, 5, 7]      # regularization
}

rand_search_cat = RandomizedSearchCV(
    CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=0,
        cat_features=cat_features
    ),
    param_distributions=param_dist_cat,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_cat.fit(X_train_cat, y_train)

print("\nBest CatBoost (Tuned, no weights) params:", rand_search_cat.best_params_)
print("Best CV F1 (CatBoost tuned):", rand_search_cat.best_score_)

cat_results = evaluate_model(
    "CatBoost - Tuned (no weights)",
    rand_search_cat.best_estimator_,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features,
    is_tuned=True,
    cv_search_object=rand_search_cat
)

# ---------------------------
# 4) CatBoost Weighted + Tuned
# ---------------------------
rand_search_cat_weighted = RandomizedSearchCV(
    CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=0,
        class_weights=[1, 2],
        cat_features=cat_features
    ),
    param_distributions=param_dist_cat,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_cat_weighted.fit(X_train_cat, y_train)

print("\nBest CatBoost (Weighted + Tuned) params:", rand_search_cat_weighted.best_params_)
print("Best CV F1 (CatBoost Tuned + Weighted):", rand_search_cat_weighted.best_score_)

cat_results = evaluate_model(
    "CatBoost - Weighted + Tuned",
    rand_search_cat_weighted.best_estimator_,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features,
    is_tuned=True,
    cv_search_object=rand_search_cat_weighted
)

# ---------------------------
# Summary + Visualization
# ---------------------------
df_cat_results = pd.DataFrame(cat_results)
print("\n===== CatBoost Experiments Summary =====")
print(df_cat_results)

# Save experiment results
df_cat_results.to_csv("results/catboost_experiments.csv", index=False)

# Save best model details
best_cat_model = df_cat_results.loc[df_cat_results["F1 (Test)"].idxmax()]
best_cat_model.to_csv("results/catboost_best_model.csv")

plot_cols = ["Train Accuracy", "Test Accuracy",
             "Train Precision", "Test Precision",
             "Train Recall", "Test Recall",
             "Train F1", "F1 (Test)", "Specificity (Test)", "ROC-AUC (Test)"]

available_cols = [c for c in df_cat_results.columns if c in plot_cols]
df_plot = df_cat_results.set_index("Model")[available_cols]

plt.figure(figsize=(14,6))
df_plot.plot(kind='bar')
plt.ylim(0,1)
plt.title("CatBoost Variants - Train/Test Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "CatBoost_variants_metrics_comparison.png"))
plt.close()

save_best_result(cat_results, "CatBoost - Baseline", RESULTS_DIR)
save_best_result(cat_results, "CatBoost - Weighted", RESULTS_DIR)
save_best_result(cat_results, "CatBoost - Tuned (no weights)", RESULTS_DIR, search_obj=rand_search_cat)
save_best_result(cat_results, "CatBoost - Weighted + Tuned", RESULTS_DIR, search_obj=rand_search_cat_weighted)


# =====================================
# SAVE BEST CATBOOST MODEL + METRICS
# =====================================
import joblib, os, json

# Create directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_metrics", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Best CatBoost = Weighted + Tuned (from experiments above)
best_cat_model = rand_search_cat_weighted.best_estimator_
joblib.dump(best_cat_model, "saved_models/catboost_weighted_tuned.pkl")

# Use last CatBoost result as metrics
cat_metrics = cat_results[-1].copy()
cat_metrics["train_columns"] = X_train_cat.columns.tolist()
cat_metrics["CV Mean F1"] = rand_search_cat_weighted.best_score_
joblib.dump(cat_metrics, "saved_metrics/catboost_weighted_tuned_metrics.pkl")

print("✅ CatBoost model saved to saved_models/catboost_weighted_tuned.pkl")
print("✅ Metrics saved to saved_metrics/catboost_weighted_tuned_metrics.pkl")


# =====================================
# SAVE ALL RESULTS HISTORY
# =====================================
with open("results/log_results.json", "w") as f:
    json.dump(log_results, f, indent=4)
logger.info("✅ Saved all Logistic Regression results to results/log_results.json")

with open("results/rf_results.json", "w") as f:
    json.dump(rf_results, f, indent=4)
logger.info("✅ Saved all Random Forest results to results/rf_results.json")

with open("results/cat_results.json", "w") as f:
    json.dump(cat_results, f, indent=4)
logger.info("✅ Saved all CatBoost results to results/cat_results.json")


# =====================================
# LEADERBOARD (Best Models Only)
# =====================================
best_log_result = next((res for res in reversed(log_results) if 'Tuned' in res.get('Model', '')), None)
best_rf_result = next((res for res in reversed(rf_results) if 'Tuned' in res.get('Model', '')), None)
best_cat_result = next((res for res in reversed(cat_results) if 'Weighted + Tuned' in res.get('Model', '')), None)

leaderboard_data = []
for best_model in [best_log_result, best_rf_result, best_cat_result]:
    if best_model:
        leaderboard_data.append(best_model)

with open("results/best_models.json", "w") as f:
    json.dump(leaderboard_data, f, indent=4)
logger.info("✅ Saved best model results to results/best_models.json")

print("\n=== Leaderboard Saved ===")
print(json.dumps(leaderboard_data, indent=4))


