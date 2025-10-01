# --------------------------------
# Block: Import Libraries
# --------------------------------

# ---------------- Core Libraries ----------------
import pandas as pd  # For data tables (like Excel in code)
import numpy as np  # For math operations (numbers and arrays)

# ---------------- Visualization ----------------
import seaborn as sns  # For pretty charts
import matplotlib.pyplot as plt  # For basic plots
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# ---------------- Scikit-learn Tools ----------------
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data into groups
from sklearn.impute import SimpleImputer  # For filling missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For converting categories to numbers and scaling
from sklearn.compose import ColumnTransformer  # To apply different steps to different columns
from sklearn.pipeline import Pipeline  # To chain steps together
import joblib  # For saving the pipeline

# ---------------- Models ----------------
from sklearn.linear_model import LogisticRegression  # For Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # For Random Forest model
from xgboost import XGBClassifier  # For XGBoost model
from catboost import CatBoostClassifier  # For CatBoost model

# ---------------- Evaluation ----------------
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    RocCurveDisplay,  # Import RocCurveDisplay to avoid NameError
)

# ---------------- Imbalanced Data Handling ----------------
from imblearn.over_sampling import SMOTE  # For handling imbalanced data

# ---------------- Hyperparameter Search ----------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# --------------------------------
# Block: Define File Paths and Constants
# --------------------------------
# Storytelling: Now that our kitchen is stocked, let's label our jars and set some rules. We're noting where our raw ingredients (data) are stored, where we'll put the finished dish, a magic number to make our experiments repeatable like planting the same seeds in a garden, and the key thing we're trying to predict – whether there's an insurance claim or not.

# File paths and constants
# Layman: These are like labels for where your data files are stored and key names.
RAW_PATH = "Car_Insurance_Claim.csv"  # Path to your raw CSV file
CLEANED_PATH = "Car_Insurance_Claim_cleaned.csv"  # Where we'll save the cleaned version
RANDOM_STATE = 42  # A fixed number to make results repeatable (like setting a seed for a garden)
TARGET_COL = "OUTCOME"  # The column we're predicting (0=no claim, 1=claim)

# --------------------------------
# Block: 1. Load & Backup Raw Data
# --------------------------------
# Storytelling: Let's open the cookbook! We load our raw data from the file into a table, make a safe copy so we don't mess up the original, and peek at its size to know how big our meal will be.

# What: Read the CSV file into a table (DataFrame) and make a copy.
# Why: We work on the copy to keep the original safe.
# Layman: Like opening a cookbook and photocopying it before marking it up.
df_raw = pd.read_csv(RAW_PATH)  # Load the raw data from the file

import os
import pandas as pd

# Define target column
TARGET_COL = "OUTCOME"

# Create directory to save plots
PLOT_DIR = "plots/eda"
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------------
# EDA REPORT FUNCTION
# --------------------------------
def quick_report(df, name="dataset"):
    print(f"\nReport for {name}:")

    # Shape
    print("Shape:", df.shape)

    # Top rows
    print("\nTop 5 rows:")
    print(df.head())

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")
    print(df.isna().sum())

    # Target distribution
    if TARGET_COL in df.columns:
        print("\nTarget distribution (counts):")
        print(df[TARGET_COL].value_counts())
        print("\nTarget distribution (proportions):")
        print(df[TARGET_COL].value_counts(normalize=True))
    else:
        print(f"⚠️ Target column '{TARGET_COL}' not found in {name}")

    # Numeric summary
    print("\nNumeric Columns Summary:")
    print(df.select_dtypes(include=['number']).describe().T)

    # Categorical summary
    print("\nCategorical Columns Summary:")
    print(df.select_dtypes(include=['object', 'category']).describe().T)

    # Unique values per column
    print("\nUnique Values per Column:")
    print(df.nunique())

    # Top categories per categorical column
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(f"\nColumn: {col}")
        print(f"Total categories: {df[col].nunique()}")
        print(df[col].value_counts(normalize=True).head(10))

    print("-" * 50)

# Run quick report
quick_report(df_raw, name="Raw Dataset")

import matplotlib.pyplot as plt
import seaborn as sns

# Select all numeric columns from the raw dataframe
numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
# Exclude the target column from the numeric columns
numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

# Loop through each numeric column to create visualizations
for col in numeric_cols:
    # ---------------------------
    # Boxplot: shows distribution and outliers by target class
    # ---------------------------
    plt.figure(figsize=(6,4))
    sns.boxplot(x=TARGET_COL, y=col, data=df_raw)
    plt.title(f"{col} vs {TARGET_COL}")  # Title indicating which column vs target
    plt.tight_layout()
    # Save the boxplot to the PLOT_DIR
    plt.savefig(os.path.join(PLOT_DIR, f"box_{col}_vs_{TARGET_COL}.png"))
    plt.close()  # Display the boxplot

    # ---------------------------
    # Histogram: shows density distribution by target class
    # ---------------------------
    plt.figure(figsize=(6,4))
    sns.histplot(
        data=df_raw,
        x=col,
        hue=TARGET_COL,      # Separate distributions by target class
        kde=True,             # Add kernel density estimate
        element="step",       # Step plot style
        stat="density"        # Normalize to density
    )
    plt.title(f"{col} distribution by {TARGET_COL}")  # Title indicating distribution
    plt.tight_layout()
    # Save the histogram to the PLOT_DIR
    plt.savefig(os.path.join(PLOT_DIR, f"hist_{col}_by_{TARGET_COL}.png"))
    plt.close()  # Display the histogram

# --------------------------------
# 1. Define target and features, split dataset
# --------------------------------
print("1️⃣ Defining target and features, performing train/test split...")

TARGET_COL = "OUTCOME"  # Uppercase as in raw dataset
features = [c for c in df_raw.columns if c != TARGET_COL]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_raw[features], df_raw[TARGET_COL],
    test_size=0.2,
    random_state=42,
    stratify=df_raw[TARGET_COL]
)

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

X_train_raw = X_train.copy()  # right after splitting, before cleaning

# --------------------------------
# 2. Clean column names
# --------------------------------
print("2️⃣ Cleaning column names...")

def clean_column_names(df):
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
    return df

X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)

TARGET_COL = TARGET_COL.lower() if TARGET_COL.lower() in df_raw.columns else TARGET_COL


print("✅ Column names cleaned. Sample columns:", X_train.columns.tolist()[:10])

# --------------------------------
# 3. Standardize categorical formats
# --------------------------------
print("3️⃣ Standardizing categorical columns (strip spaces, lowercase)...")

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).str.strip().str.lower()
    X_test[col] = X_test[col].astype(str).str.strip().str.lower()

print(f"✅ Standardized {len(categorical_cols)} categorical columns: {categorical_cols}")

# --------------------------------
# 4. Drop columns not needed
# --------------------------------
print("4️⃣ Dropping unnecessary columns...")

cols_to_drop = ['id', 'race', 'vehicle_type', 'postal_code']
cols_to_drop_existing = [c for c in cols_to_drop if c in X_train.columns]

X_train = X_train.drop(columns=cols_to_drop_existing)
X_test = X_test.drop(columns=cols_to_drop_existing)

print(f"✅ Dropped columns: {cols_to_drop_existing}")

# --------------------------------
# 5. Outlier detection & clipping
# --------------------------------
print("5️⃣ Detecting and clipping outliers in numeric columns...")

numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

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
# 6. Impute Missing Values (Corrected)
# --------------------------------
print("6️⃣ Imputing missing values...")

from sklearn.impute import SimpleImputer

# Recompute numeric and categorical columns after dropping unnecessary columns
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# Numeric
num_imputer = SimpleImputer(strategy='median')
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

# Categorical
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

print(f"✅ Missing values imputed for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.")
print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# --------------------------------
# 7. Feature Engineering: Risk Score
# --------------------------------
print("7️⃣ Creating RISK_SCORE from violations/accidents if present...")

risk_components = ['speeding_violations', 'duis', 'past_accidents']
existing_risk_cols = [c for c in risk_components if c in X_train.columns]

if existing_risk_cols:
    X_train['risk_score'] = X_train[existing_risk_cols].sum(axis=1)
    X_test['risk_score'] = X_test[existing_risk_cols].sum(axis=1)

    X_train = X_train.drop(columns=existing_risk_cols)
    X_test = X_test.drop(columns=existing_risk_cols)
    print(f"✅ Created RISK_SCORE and dropped columns: {existing_risk_cols}")
else:
    print("⚠️ No risk score components found in the data.")

# --------------------------------
# 8. Prepare unencoded CatBoost data
# --------------------------------
print("8️⃣ Preparing unencoded data for CatBoost...")

X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

cat_features = ['age', 'gender', 'driving_experience', 'education', 'income', 'vehicle_year']
numeric_cols_cat = [c for c in X_train_cat.columns if c not in cat_features]

print(f"✅ CatBoost categorical features: {cat_features}")
print(f"✅ Numeric features for CatBoost: {numeric_cols_cat}")

# --------------------------------
# 9. Quick report after cleaning & feature engineering
# --------------------------------
print("9️⃣ Generating final quick report...")

def quick_report(df, name="dataset"):
    print(f"\nReport for {name}:")
    print("Shape:", df.shape)
    print("\nTop 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.select_dtypes(include=['number']).describe().T)
    print("\nCategorical summary:")
    print(df.select_dtypes(include=['object', 'category']).describe().T)
    print("\nUnique values per column:")
    print(df.nunique())
    print("-"*40)

quick_report(X_train, name="X_train cleaned")
quick_report(X_test, name="X_test cleaned")

import os
import joblib

# Create a directory for cleaned data
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save cleaned training/testing features and targets
joblib.dump(X_train, os.path.join(SAVE_DIR, "X_train_cleaned.pkl"))
joblib.dump(X_test, os.path.join(SAVE_DIR, "X_test_cleaned.pkl"))
joblib.dump(y_train, os.path.join(SAVE_DIR, "y_train.pkl"))
joblib.dump(y_test, os.path.join(SAVE_DIR, "y_test.pkl"))

# Optionally save the raw dataframe
joblib.dump(df_raw, os.path.join(SAVE_DIR, "df_raw.pkl"))

print(f"✅ All cleaned datasets saved in '{SAVE_DIR}'")

import matplotlib.pyplot as plt
import seaborn as sns

# Create directory for plots
PLOT_DIR = "plots after cleaning"
os.makedirs(PLOT_DIR, exist_ok=True)

print("2A️⃣ Numeric Features: Before vs After Cleaning")

numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(X_train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col} (After Cleaning)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_after_cleaning.png"))
    plt.close()

print("2B️⃣ Categorical Features: After Cleaning")

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    # Use x=col to make bars vertical (upright)
    sns.countplot(x=col, data=X_train, order=X_train[col].value_counts().index)
    plt.title(f"Distribution of {col} (After Cleaning)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks for readability
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_after_cleaning.png"))
    plt.close()

print("2C️⃣ Target Distribution After Cleaning")

plt.figure(figsize=(6,4))
sns.countplot(x=TARGET_COL, data=X_train.join(y_train))
plt.title("Target Distribution (Training Set After Cleaning)")
plt.xlabel(TARGET_COL)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{TARGET_COL}_distribution_after_cleaning.png"))
plt.close()

print("2D️⃣ Numeric Features: Boxplot After Cleaning")

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=X_train[col])
    plt.title(f"Boxplot of {col} (After Cleaning/Clipping)")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_boxplot_after_cleaning.png"))
    plt.close()

# Recompute numeric_cols based on the cleaned X_train DataFrame as this is used for iterating
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Map cleaned names back to raw names for plotting
# Create a dictionary to map cleaned names to raw names
cleaned_to_raw_map = {col.lower().replace(" ", "_"): col for col in X_train_raw.columns}
# Exclude 'risk_score' from the list of columns to get from X_train_raw
numeric_cols_raw = [cleaned_to_raw_map.get(col, col) for col in numeric_cols if col != 'risk_score']

for i, col in enumerate(numeric_cols):
    plt.figure(figsize=(10,4))
    # Use the corresponding raw column name for X_train_raw if it exists, otherwise skip the raw plot for this column
    if col != 'risk_score':
        sns.kdeplot(X_train_raw[numeric_cols_raw[i]], color='red', label='Before Cleaning', fill=True, alpha=0.3)
    sns.kdeplot(X_train[col], color='blue', label='After Cleaning', fill=True, alpha=0.3)
    plt.title(f"Distribution of {col}: Before vs After Cleaning")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_comparison.png"))
    plt.close()

print("Columns in X_train:", X_train.columns.tolist())
print("Columns in X_test:", X_test.columns.tolist())

print("y_train unique values:", y_train.unique())
print("y_test unique values:", y_test.unique())

print("\n=== TARGET SEPARATION CHECK ===")
# Checking if the number of rows in the training features matches the number of rows in the training target
print("Train features vs target rows equal?", X_train.shape[0] == y_train.shape[0])
# Checking if the number of rows in the test features matches the number of rows in the test target
print("Test features vs target rows equal?", X_test.shape[0] == y_test.shape[0])
# Displaying the first 10 feature column names from the training set for a quick overview
print("Sample feature columns:\n", X_train.columns[:10])
# Displaying the frequency distribution of the training target values to check class balance
print("Sample target values:\n", y_train.value_counts())

from sklearn.preprocessing import OneHotEncoder

# Define numeric and categorical columns
numeric_cols = ['credit_score', 'vehicle_ownership', 'married', 'children', 'annual_mileage', 'risk_score']
categorical_cols = ['age', 'gender', 'driving_experience', 'education', 'income', 'vehicle_year']

# One-hot encode categorical columns
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit on training data
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat  = ohe.transform(X_test[categorical_cols])

# Convert to DataFrame and merge with numeric columns
X_train_encoded = pd.concat([
    X_train[numeric_cols].reset_index(drop=True),
    pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_cols))
], axis=1)

X_test_encoded = pd.concat([
    X_test[numeric_cols].reset_index(drop=True),
    pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(categorical_cols))
], axis=1)

print("✅ Encoded feature shape (train):", X_train_encoded.shape)
print("✅ Encoded feature shape (test):", X_test_encoded.shape)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit on train, transform train and test
X_train_rf_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_rf_cat = ohe.transform(X_test[categorical_cols])

# Convert back to DataFrame with proper column names
encoded_cols = ohe.get_feature_names_out(categorical_cols)
X_train_rf_cat = pd.DataFrame(X_train_rf_cat, columns=encoded_cols, index=X_train.index)
X_test_rf_cat = pd.DataFrame(X_test_rf_cat, columns=encoded_cols, index=X_test.index)

numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

X_train_rf = pd.concat([X_train[numeric_cols], X_train_rf_cat], axis=1)
X_test_rf = pd.concat([X_test[numeric_cols], X_test_rf_cat], axis=1)

print("Final X_train shape for RandomForest:", X_train_rf.shape)
print("Final X_test shape for RandomForest:", X_test_rf.shape)

# --------------------------------
# Function: evaluate_model
# What: Evaluate classifier on train/test sets, compute full metrics, generate plots.
# Why: Standardize evaluation and visualizations for all models.
# --------------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns

def evaluate_model(
    model_name,
    model,
    X_train, y_train,
    X_test, y_test,
    results_list,
    plot=True,
    cat_features=None,
    is_tuned=False,
    cv_search_object=None
):
    print(f"\nEvaluating model: {model_name}")

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:  # fallback
        y_test_prob = y_test_pred

    # Compute metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_test_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Print metrics
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Train Precision: {train_prec:.4f}, Test Precision: {test_prec:.4f}")
    print(f"Train Recall: {train_rec:.4f}, Test Recall: {test_rec:.4f}")
    print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    print(f"Specificity (Test): {specificity:.4f}, ROC-AUC (Test): {roc_auc:.4f}")

    # Save results
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
        "ROC-AUC (Test)": roc_auc
    }

    results_list.append(result_dict)

    # ---------------------------
    # Plots
    # ---------------------------
    if plot:
        # Confusion matrix
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
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve (Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_roc_curve.png"))
        plt.close()

    return results_list

# --------------------------------
# Block: Random Forest Experiments (4 Variants with SMOTE & Tuning)
# --------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import os

rf_results = []
PLOT_DIR = "plots/randomforest"
os.makedirs(PLOT_DIR, exist_ok=True)

# Use the encoded data for Random Forest
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

# ---------------------------
# Summary DataFrame + Bar Chart
# ---------------------------
df_rf_results = pd.DataFrame(rf_results)
print("\n===== Random Forest Experiments Summary =====")
print(df_rf_results)

plot_cols = ["Train Accuracy", "Test Accuracy",
             "Train Precision", "Test Precision",
             "Train Recall", "Test Recall",
             "Train F1", "F1 (Test)", "Specificity (Test)", "ROC-AUC (Test)"]

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

# --------------------------------
# Block: Logistic Regression Experiments (4 Variants with Scaling)
# --------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import os

log_results = []
PLOT_DIR = "plots/logreg"
os.makedirs(PLOT_DIR, exist_ok=True)

# Use the encoded data for Logistic Regression
X_train_lr = X_train_encoded
X_test_lr = X_test_encoded

# Scale numeric features for Logistic Regression
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

# ---------------------------
# 3) Logistic Regression - Tuned (No Weights)
# ---------------------------
param_dist_log = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2", None],  # saga supports only l1, l2, elasticnet, none
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

# ---------------------------
# Summary DataFrame + Bar Chart
# ---------------------------
df_log_results = pd.DataFrame(log_results)
print("\n===== Logistic Regression Experiments Summary =====")
print(df_log_results)

plot_cols = ["Train Accuracy", "Test Accuracy",
             "Train Precision", "Test Precision",
             "Train Recall", "Test Recall",
             "Train F1", "F1 (Test)", "ROC-AUC (Test)"]

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
cat_results = []
PLOT_DIR = "plots/catboost"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------
# Define numeric and categorical features based on cleaned data
# ---------------------------
numeric_cols = ['credit_score', 'vehicle_ownership', 'married', 'children', 'annual_mileage', 'risk_score']
cat_features = ['age', 'gender', 'driving_experience', 'education', 'income', 'vehicle_year']


# Use the cleaned, unencoded data
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

# ---------------------------
# 1) CatBoost Baseline
# ---------------------------
cat_baseline = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, cat_features=cat_features) # Pass cat_features
cat_results = evaluate_model(
    "CatBoost - Baseline",
    cat_baseline,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features # Pass cat_features to evaluate_model
)

# ---------------------------
# 2) CatBoost Weighted
# ---------------------------
cat_weighted = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, class_weights=[1, 2], cat_features=cat_features) # Pass cat_features
cat_results = evaluate_model(
    "CatBoost - Weighted",
    cat_weighted,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features # Pass cat_features to evaluate_model
)

# ---------------------------
# 3) CatBoost Tuned (No Weights)
# ---------------------------
param_dist_cat = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 500, 800],
    'l2_leaf_reg': [1, 3, 5, 7]
}

rand_search_cat = RandomizedSearchCV(
    CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, cat_features=cat_features), # Pass cat_features
    param_distributions=param_dist_cat,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_cat.fit(X_train_cat, y_train) # cat_features is passed via the estimator

print("\nBest CatBoost (Tuned, no weights) params:", rand_search_cat.best_params_)
print("Best CV F1 (CatBoost tuned):", rand_search_cat.best_score_)

cat_results = evaluate_model(
    "CatBoost - Tuned (no weights)",
    rand_search_cat.best_estimator_,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features, # Pass cat_features to evaluate_model
    is_tuned=True,
    cv_search_object=rand_search_cat
)

# ---------------------------
# 4) CatBoost Weighted + Tuned
# ---------------------------
rand_search_cat_weighted = RandomizedSearchCV(
    CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, class_weights=[1, 2], cat_features=cat_features), # Pass cat_features
    param_distributions=param_dist_cat,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rand_search_cat_weighted.fit(X_train_cat, y_train) # cat_features is passed via the estimator

print("\nBest CatBoost (Weighted + Tuned) params:", rand_search_cat_weighted.best_params_)
print("Best CV F1 (CatBoost Tuned + Weighted):", rand_search_cat_weighted.best_score_)

cat_results = evaluate_model(
    "CatBoost - Weighted + Tuned",
    rand_search_cat_weighted.best_estimator_,
    X_train_cat, y_train,
    X_test_cat, y_test,
    cat_results,
    plot=True,
    cat_features=cat_features, # Pass cat_features to evaluate_model
    is_tuned=True,
    cv_search_object=rand_search_cat_weighted
)

# ---------------------------
# Summary + Visualization
# ---------------------------
df_cat_results = pd.DataFrame(cat_results)
print("\n===== CatBoost Experiments Summary =====")
print(df_cat_results)

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

cat_metrics = cat_results[-1]  # Last evaluated model, i.e., Weighted + Tuned

import joblib
import os

# Create directories if they don't exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_metrics", exist_ok=True)

# Save the final, best-tuned model (The model object itself)
joblib.dump(rand_search_cat_weighted.best_estimator_, "saved_models/catboost_weighted_tuned.pkl")

# Prepare and save the metrics dictionary
# NOTE: Ensure 'cat_metrics' is a dictionary containing your results from the leaderboard row
# We add the columns used for training to this dictionary for reproducibility.
cat_metrics["train_columns"] = X_train_cat.columns.tolist()
cat_metrics["CV Mean F1"] = rand_search_cat_weighted.best_score_
joblib.dump(cat_metrics, "saved_metrics/catboost_weighted_tuned_metrics.pkl")

print("CatBoost model saved to saved_models/catboost_weighted_tuned.pkl")
print("Metrics saved to saved_metrics/catboost_weighted_tuned_metrics.pkl")

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# ---------------------------
# Select best models from each algorithm
# ---------------------------
best_log_result = next((res for res in reversed(log_results) if 'Tuned' in res.get('Model', '')), None)
best_rf_result = next((res for res in reversed(rf_results) if 'Tuned' in res.get('Model', '')), None)
best_cat_result = next((res for res in reversed(cat_results) if 'Weighted + Tuned' in res.get('Model', '')), None)

# ---------------------------
# Build leaderboard data
# ---------------------------
leaderboard_data = []
for best_model in [best_log_result, best_rf_result, best_cat_result]:
    if best_model:
        leaderboard_data.append(best_model)

# ---------------------------
# Create DataFrame
# ---------------------------
leaderboard = pd.DataFrame(leaderboard_data)

# ---------------------------
# Automatically detect numeric columns for plotting
# ---------------------------
# Keep "Model" column for labeling
metric_cols = [col for col in leaderboard.columns if col != "Model"]

# Convert all metric columns to numeric
for col in metric_cols:
    leaderboard[col] = pd.to_numeric(leaderboard[col], errors='coerce')

# Rank by F1 (test) if available
if "F1 (test)" in leaderboard.columns:
    leaderboard["Rank (by F1 Test)"] = leaderboard["F1 (test)"].rank(ascending=False, method='min')

# ---------------------------
# Display leaderboard
# ---------------------------
print("\n=== Leaderboard: Final Models ===")
print(leaderboard.sort_values("Rank (by F1 Test)") if "Rank (by F1 Test)" in leaderboard.columns else leaderboard)

# ---------------------------
# Plot leaderboard metrics dynamically
# ---------------------------
if not leaderboard.empty:
    # All numeric metrics except "Rank" for plotting
    plot_metrics = [col for col in metric_cols if col in leaderboard.columns and col != "Rank (by F1 Test)"]

    df_plot = leaderboard.set_index("Model")[plot_metrics]

    ax = df_plot.plot(kind='bar', figsize=(14,7))
    plt.ylim(0, 1.05)
    plt.title("Leaderboard: Best Models Comparison (All Metrics)")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/leaderboard_comparison_dynamic.png")
    plt.show()

