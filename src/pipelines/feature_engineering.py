import os
import pandas as pd
import numpy as np
import yaml

def load_config():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(repo_root, "config", "config.yaml")

    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            with open(config_path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_behavioral_features():
    config = load_config()
    raw_path = config["paths"]["raw_data_path"]
    processed_path = config["paths"]["processed_data_path"]
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw transaction stream not found at {raw_path}. Run the generator first!")
        
    print(f"[Feature Pipeline] Parsing raw transaction logs from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Convert string timestamps to datetime objects for temporal calculations
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate transaction weight: + for deposits, - for withdrawals, 0 for logs/errors
    df["tx_effect"] = df.apply(
        lambda row: row["amount"] if row["event_type"] == "deposit" 
        else (-row["amount"] if row["event_type"] == "withdrawal" else 0), axis=1
    )
    
    features_list = []
    
    # Group logs by unique customer to calculate individual behavioral shifts
    for customer_id, group in df.groupby("customer_id"):
        group = group.sort_values("timestamp")
        
        # Calculate running balance starting from an arbitrary $5000 corporate baseline
        running_balance = 5000.0 + group["tx_effect"].cumsum()
        current_balance = running_balance.iloc[-1]
        peak_balance = running_balance.max()
        
        # Metric 1: Wallet Depletion Ratio (Has their balance dropped significantly from its peak?)
        balance_to_max_ratio = round(current_balance / peak_balance, 4) if peak_balance > 0 else 0.0
        
        total_tx = len(group)
        failed_swipes = len(group[group["event_type"] == "failed_swipe"])
        support_tickets = len(group[group["event_type"] == "support_ticket"])
        
        # Metric 2: Transaction Velocity Drop (Proxy slider for simulation metrics)
        tx_velocity_drop_ratio = round(np.random.uniform(0.3, 0.95), 2)
        
        features_list.append({
            "customer_id": customer_id,
            "balance_to_max_ratio": balance_to_max_ratio,
            "tx_velocity_drop_ratio": tx_velocity_drop_ratio,
            "failed_tx_count_7d": failed_swipes,
            "support_tickets_30d": support_tickets,
            "is_active_credit_card_user": 1 if total_tx > 3 else 0
        })
        
    features_df = pd.DataFrame(features_list)
    
    # --- INJECTING REALISTIC PROBABILISTIC LABELS ---
    # 1. Base risk score computation derived from the raw friction markers
    risk_score = (
        (features_df['failed_tx_count_7d'] * 0.30) + 
        (features_df['support_tickets_30d'] * 0.25) + 
        ((1.0 - features_df['balance_to_max_ratio']) * 0.25)
    )
    
    # 2. Add random human variance noise (stochasticity) so the model cannot perfectly memorize rules
    np.random.seed(config["model_parameters"]["random_state"])
    noise = np.random.uniform(-0.15, 0.15, len(features_df))
    final_probability = np.clip(risk_score + noise, 0.0, 1.0)
    
    # 3. Assign final binary target based on probability threshold
    features_df['churned'] = (final_probability > 0.45).astype(int)
    # ------------------------------------------------
    
    # Ensure processed directory path exists securely
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    features_df.to_csv(processed_path, index=False)
    print(f"[Feature Pipeline] Feature matrix successfully updated at: {processed_path}")

if __name__ == "__main__":
    build_behavioral_features()