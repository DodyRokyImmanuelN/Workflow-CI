import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
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
# CONFIGURATION
# =====================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =====================================================
# PARAMETER HANDLING
# =====================================================

def get_parameters():
    """
    Ambil parameter dari environment variable
    (MLflow Project akan inject parameter di sini)
    """
    return {
        "data_path": os.getenv("DATA_PATH", "heart_preprocessing.csv"),
        "model_type": os.getenv("MODEL_TYPE", "RandomForest"),
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
    print(f"📂 Loading data from: {path}")

    search_paths = [
        path,
        f"./{path}",
        f"../{path}",
        f"preprocessing/{path}",
        f"../preprocessing/{path}"
    ]

    for p in search_paths:
        if os.path.exists(p):
            print(f"✅ Found dataset at: {p}")
            df = pd.read_csv(p)
            print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

    raise FileNotFoundError(f"❌ Dataset not found. Tried: {search_paths}")

# =====================================================
# PREPARE DATA
# =====================================================

def prepare_data(df, test_size, random_state):
    print("\n🔄 Preparing data...")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"✅ Train size: {len(X_train)}")
    print(f"✅ Test size : {len(X_test)}")
    print(f"✅ Features : {list(X.columns)}")

    return X_train, X_test, y_train, y_test

# =====================================================
# TRAIN MODEL (MLFLOW PROJECT SAFE)
# =====================================================

def train_model(X_train, X_test, y_train, y_test, params):
    print("\n🤖 Training model with MLflow...")

    # Autolog TANPA start_run (WAJIB untuk MLflow Project)
    mlflow.sklearn.autolog(log_models=True)

    # Ambil run_id dari environment (CARA BENAR)
    run_id = os.environ.get("MLFLOW_RUN_ID", "unknown")
    print(f"🔗 MLflow Run ID: {run_id}")

    model_params = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "random_state": params["random_state"],
        "n_jobs": -1
    }

    print(f"📋 Model parameters: {model_params}")

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "run_id": run_id,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

    # Simpan run id (optional)
    with open("latest_run_id.txt", "w") as f:
        f.write(run_id)

    return model, y_pred, y_test, X_test, metrics

# =====================================================
# SAVE VISUALIZATIONS
# =====================================================

def save_visualizations(model, X_test, y_test, y_pred):
    print("\n📊 Saving visualizations...")
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(importances)),
        importances[indices]
    )
    plt.xticks(
        range(len(importances)),
        X_test.columns[indices],
        rotation=45,
        ha="right"
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance.png")
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)

# =====================================================
# SAVE MODEL
# =====================================================

def save_model(model):
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/heart_disease_model.pkl"
    joblib.dump(model, path)
    print(f"✅ Model saved: {path}")
    return path

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=" * 70)
    print("🚀 MLPROJECT - HEART DISEASE CLASSIFICATION")
    print("=" * 70)
    print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print("=" * 70)

    try:
        params = get_parameters()

        print("\n📋 Parameters:")
        for k, v in params.items():
            print(f"   {k:20s}: {v}")

        df = load_data(params["data_path"])

        X_train, X_test, y_train, y_test = prepare_data(
            df,
            params["test_size"],
            params["random_state"]
        )

        model, y_pred, y_test_actual, X_test_actual, metrics = train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            params
        )

        save_visualizations(model, X_test_actual, y_test_actual, y_pred)
        model_path = save_model(model)

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Run ID    : {metrics['run_id']}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-score  : {metrics['f1']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"Artifacts : artifacts/")
        print(f"Model     : {model_path}")
        print("=" * 70)

        sys.exit(0)

    except Exception as e:
        print("\n❌ TRAINING FAILED")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
