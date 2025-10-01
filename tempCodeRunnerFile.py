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