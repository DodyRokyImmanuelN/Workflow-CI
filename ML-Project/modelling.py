import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================


# Set MLflow tracking URI from environment or default
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file://./mlruns')

# =====================================================
# PARSE PARAMETERS
# =====================================================
def get_parameters():
    """Get parameters from environment or use defaults"""
    
    params = {
        'data_path': os.getenv('DATA_PATH', 'heart_preprocessing.csv'),
        'model_type': os.getenv('MODEL_TYPE', 'RandomForest'),
        'test_size': float(os.getenv('TEST_SIZE', '0.2')),
        'random_state': int(os.getenv('RANDOM_STATE', '42')),
        'n_estimators': int(os.getenv('N_ESTIMATORS', '100')),
        'max_depth': int(os.getenv('MAX_DEPTH', '10')),
        'min_samples_split': int(os.getenv('MIN_SAMPLES_SPLIT', '5'))
    }
    
    return params

# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    """Load preprocessed dataset"""
    print(f"📂 Loading data from: {path}")
    
    if not os.path.exists(path):
        # Try alternative paths
        alt_paths = [
            path,
            f"../{path}",
            f"preprocessing/{path}",
            f"../preprocessing/{path}"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"   ✅ Found at: {alt_path}")
                path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"❌ Data file not found: {path}\n"
                f"   Searched locations: {alt_paths}"
            )
    
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# =====================================================
# PREPARE DATA
# =====================================================
def prepare_data(df, test_size, random_state):
    """Split data into train and test sets"""
    print(f"\n🔄 Preparing data (test_size={test_size}, random_state={random_state})...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data dengan stratify untuk menjaga proporsi class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"✅ Train size: {X_train.shape[0]} samples")
    print(f"✅ Test size: {X_test.shape[0]} samples")
    print(f"✅ Class distribution (train): {dict(y_train.value_counts())}")
    print(f"✅ Class distribution (test): {dict(y_test.value_counts())}")
    print(f"✅ Features: {list(X.columns)}")
    
    return X_train, X_test, y_train, y_test

# =====================================================
# TRAIN MODEL WITH MLFLOW
# =====================================================
def train_model(X_train, X_test, y_train, y_test, params):
    print("\n🤖 Training model with MLflow...")
    
    # mlflow run SUDAH membuat run → JANGAN start run lagi
    mlflow.sklearn.autolog()

    # Model parameters
    model_params = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'random_state': params['random_state'],
        'n_jobs': -1
    }

    print(f"📋 Model parameters: {model_params}")

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    run_id = mlflow.active_run().info.run_id

    with open('latest_run_id.txt', 'w') as f:
        f.write(run_id)

    return model, y_pred, y_test, X_test, {
        'run_id': run_id,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# =====================================================
# SAVE VISUALIZATIONS
# =====================================================
def save_visualizations(model, X_test, y_test, y_pred):
    """Save visualizations as artifacts"""
    print("\n📊 Creating visualizations...")
    
    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Heart Disease', 'Heart Disease'],
               yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.title('Confusion Matrix - Heart Disease MLProject', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = 'artifacts/heart_confusion_matrix.png'
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix: {cm_path}")
    
    # Feature Importance
    feature_names = X_test.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    bars = plt.bar(range(X_test.shape[1]), importances[indices], color=colors)
    plt.title('Feature Importance - Heart Disease MLProject', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(X_test.shape[1]), 
              [feature_names[i] for i in indices], 
              rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fi_path = 'artifacts/heart_feature_importance.png'
    plt.savefig(fi_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Feature importance: {fi_path}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, 
                                  target_names=['No Heart Disease', 'Heart Disease'])
    report_path = 'artifacts/heart_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("HEART DISEASE CLASSIFICATION REPORT - MLPROJECT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "="*60 + "\n")
        f.write("FEATURE IMPORTANCE\n")
        f.write("="*60 + "\n")
        for i, idx in enumerate(indices):
            f.write(f"{i+1}. {feature_names[idx]:25s}: {importances[idx]:.4f}\n")
    print(f"   ✅ Classification report: {report_path}")

# =====================================================
# SAVE MODEL
# =====================================================
def save_model(model, path='artifacts/heart_disease_model.pkl'):
    """Save model as pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"   ✅ Model saved: {path}")
    return path

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("="*70)
    print("🚀 MLPROJECT - HEART DISEASE CLASSIFICATION TRAINING")
    print("="*70)
    print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")
    
    try:
        # Get parameters
        params = get_parameters()
        print(f"📋 Training Parameters:")
        for key, value in params.items():
            print(f"   {key:20s}: {value}")
        
        # Load & prepare data
        df = load_data(params['data_path'])
        X_train, X_test, y_train, y_test = prepare_data(
            df, params['test_size'], params['random_state']
        )
        
        # Train model with MLflow
        model, y_pred, y_test_actual, X_test_actual, metrics = train_model(
            X_train, X_test, y_train, y_test, params
        )
        
        # Save visualizations & model
        save_visualizations(model, X_test_actual, y_test_actual, y_pred)
        model_path = save_model(model)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n📊 Final Metrics:")
        print(f"   Run ID    : {metrics['run_id']}")
        print(f"   Accuracy  : {metrics['accuracy']:.4f}")
        print(f"   Precision : {metrics['precision']:.4f}")
        print(f"   Recall    : {metrics['recall']:.4f}")
        print(f"   F1-Score  : {metrics['f1']:.4f}")
        print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"\n📁 Artifacts saved in: artifacts/")
        print(f"📁 Model saved at: {model_path}")
        print(f"\n🔗 View results: mlflow ui")
        print("="*70 + "\n")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)