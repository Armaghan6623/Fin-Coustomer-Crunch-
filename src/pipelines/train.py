import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import joblib

# Determine absolute path thresholds dynamically
# This explicitly finds the repository root folder, no matter who triggers the script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/pipelines
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # hunnoia/

def load_config():
    config_path = os.path.join(REPO_ROOT, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_model_training():
    config = load_config()
    
    # Anchor paths to REPO_ROOT cleanly
    processed_data_path = os.path.abspath(os.path.join(REPO_ROOT, config["paths"]["processed_data_path"]))
    relative_model_path = config["paths"]["model_output_path"]
    model_output_path = os.path.abspath(os.path.join(REPO_ROOT, relative_model_path))
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed feature matrix missing at {processed_data_path}. Run feature engineering first!")
        
    print(f"[Model Training] Loading processed feature store from: {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # 1. Separate Features (X) and Target Label (y)
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]
    
    print(f"[Model Training] Dataset summary - Total samples: {len(df)}, Total features: {X.shape[1]}")
    print(f"[Model Training] Target distribution class balance: \n{y.value_counts()}")
    
    # 2. Perform a Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["model_parameters"]["test_size"], 
        random_state=config["model_parameters"]["random_state"],
        stratify=y
    )
    
    print("[Model Training] Training an Imbalance-Aware RandomForest Engine...")
    
    # 3. Initialize and train the model using class weighting
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=config["model_parameters"]["random_state"],
        class_weight="balanced"  
    )
    model.fit(X_train, y_train)
    
    # 4. Run Model Evaluation
    print("\n" + "="*50 + "\n[Model Evaluation] Performance Telemetry Summary:\n" + "="*50)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # Calculate Precision-Recall Area Under Curve (PR-AUC)
    probabilities = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = auc(recall, precision)
    print(f"Calculated Precision-Recall AUC (PR-AUC): {round(pr_auc, 4)}")
    
    # 5. Persist/Serialize the Model to disk for deployment
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\n[Model Training] Champion model binary saved successfully to absolute path: {model_output_path}")

if __name__ == "__main__":
    run_model_training()