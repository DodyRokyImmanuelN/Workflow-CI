import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # CI-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 🔴 WAJIB: SET EXPERIMENT DI TOP-LEVEL
# =====================================================
EXPERIMENT_NAME = "HeartDisease_Classification_MLProject"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================================================
# PARAMETERS
# =====================================================
def get_parameters():
    return {
        "data_path": os.getenv("DATA_PATH", "heart_preprocessing.csv"),
        "test_size": float(os.getenv("TEST_SIZE", "0.2")),
        "random_state": int(os.getenv("RANDOM_STATE", "42")),
        "n_estimators": int(os.getenv("N_ESTIMATORS", "100")),
        "max_depth": int(os.getenv("MAX_DEPTH", "10")),
        "min_samples_split": int(os.getenv("MIN_SAMPLES_SPLIT", "5")),
    }

# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    print(f"📂 Loading data: {path}")

    if not os.path.exists(path):
        alt_paths = [
            path,
            f"../{path}",
            f"preprocessing/{path}",
            f"../preprocessing/{path}",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape}")
    return df

# =====================================================
# PREPARE DATA
# =====================================================
def prepare_data(df, test_size, random_state):
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

# =====================================================
# TRAIN MODEL (MLFLOW PROJECT)
# =====================================================
def train_model(X_train, X_test, y_train, y_test, params):
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="MLProject_HeartDisease_Training"):

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")

        run_id = mlflow.active_run().info.run_id
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)

        print(f"🔗 Run ID: {run_id}")
        return model, y_pred, metrics, run_id

# =====================================================
# SAVE ARTIFACTS
# =====================================================
def save_artifacts(model, X_test, y_test, y_pred):
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/heart_confusion_matrix.png")
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), X_test.columns[idx], rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("artifacts/heart_feature_importance.png")
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open("artifacts/heart_classification_report.txt", "w") as f:
        f.write(report)

    # Save model
    joblib.dump(model, "artifacts/heart_disease_model.pkl")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 MLFLOW PROJECT – HEART DISEASE TRAINING")
    print("=" * 70)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print("=" * 70)

    try:
        params = get_parameters()
        df = load_data(params["data_path"])

        X_train, X_test, y_train, y_test = prepare_data(
            df, params["test_size"], params["random_state"]
        )

        model, y_pred, metrics, run_id = train_model(
            X_train, X_test, y_train, y_test, params
        )

        save_artifacts(model, X_test, y_test, y_pred)

        print("\n✅ TRAINING COMPLETE")
        print(f"Run ID: {run_id}")
        sys.exit(0)

    except Exception as e:
        print("❌ ERROR:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
