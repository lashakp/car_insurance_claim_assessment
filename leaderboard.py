# leaderboard.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from catboost import CatBoostClassifier

# ---------------------------
# Load results from JSON
# ---------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_json_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

log_results = load_json_results(os.path.join(RESULTS_DIR, "log_results.json"))
rf_results = load_json_results(os.path.join(RESULTS_DIR, "rf_results.json"))
cat_results = load_json_results(os.path.join(RESULTS_DIR, "cat_results.json"))

# ---------------------------
# Select best models from each algorithm
# ---------------------------
best_log_result = next((res for res in reversed(log_results) if 'Tuned' in res.get('Model', '')), None)
best_rf_result = next((res for res in reversed(rf_results) if 'Tuned' in res.get('Model', '')), None)
best_cat_result = next((res for res in reversed(cat_results) if 'Weighted + Tuned' in res.get('Model', '')), None)

# ---------------------------
# Save best models only
# ---------------------------
leaderboard_data = []
for best_model in [best_log_result, best_rf_result, best_cat_result]:
    if best_model:
        leaderboard_data.append(best_model)

with open(os.path.join(RESULTS_DIR, "best_models.json"), "w") as f:
    json.dump(leaderboard_data, f, indent=4)

print("✅ Saved best model results to results/best_models.json")
print(f"best_log_result: {best_log_result}")
print(f"best_rf_result: {best_rf_result}")
print(f"best_cat_result: {best_cat_result}")

# ---------------------------
# Load leaderboard
# ---------------------------
RESULTS_PATH = os.path.join(RESULTS_DIR, "best_models.json")
if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(
        f"{RESULTS_PATH} not found. Run training script first to generate results."
    )

with open(RESULTS_PATH, "r") as f:
    leaderboard_data = json.load(f)

leaderboard = pd.DataFrame(leaderboard_data)

# ---------------------------
# Prepare leaderboard DataFrame
# ---------------------------
metric_cols = [col for col in leaderboard.columns if col != "Model"]
for col in metric_cols:
    leaderboard[col] = pd.to_numeric(leaderboard[col], errors="coerce")

if "F1 (Test)" in leaderboard.columns:
    leaderboard["Rank (by F1 Test)"] = leaderboard["F1 (Test)"].rank(
        ascending=False, method="min"
    )

print("\n=== Leaderboard: Final Models ===")
print(
    leaderboard.sort_values("Rank (by F1 Test)")
    if "Rank (by F1 Test)" in leaderboard.columns
    else leaderboard
)

# ---------------------------
# Plot leaderboard metrics
# ---------------------------
if not leaderboard.empty:
    plot_metrics = [
        col for col in metric_cols if col in leaderboard.columns and col != "Rank (by F1 Test)"
    ]
    df_plot = leaderboard.set_index("Model")[plot_metrics]

    ax = df_plot.plot(kind="bar", figsize=(14, 7))
    plt.ylim(0, 1.05)
    plt.title("Leaderboard: Best Models Comparison (All Metrics)")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/leaderboard_comparison_dynamic.png")
    plt.show()

# ---------------------------
# Feature Importance for CatBoost
# ---------------------------
importances_dir = os.path.join("plots", "feature_importance")
os.makedirs(importances_dir, exist_ok=True)

if "CatBoost Weighted + Tuned" in leaderboard["Model"].values:
    print("\nGenerating feature importance for CatBoost...")

    # Load training data
    X_train = joblib.load("saved_data/X_train_cleaned.pkl")
    y_train = joblib.load("saved_data/y_train.pkl")

    # Load best CatBoost model
    best_cat_model = joblib.load("saved_models/catboost_weighted_tuned.pkl")

    # Compute feature importance
    feature_importance = best_cat_model.get_feature_importance()
    feature_names = X_train.columns

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    fi_df.to_csv(os.path.join(importances_dir, "catboost_feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=fi_df.head(15),
        x="Importance", y="Feature",
        palette="viridis"
    )
    plt.title("CatBoost Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(os.path.join(importances_dir, "catboost_feature_importance.png"))
    plt.close()

    print("✅ Feature importance saved in /plots/feature_importance/")
