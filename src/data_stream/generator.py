import os
import time
import random
import csv
from datetime import datetime
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

def generate_mock_transaction(customer_id: str, is_high_risk: bool) -> dict:
    """Generates events with distributions altered by the customer's risk profile."""
    
    if is_high_risk:
        # High-friction users get way more failed swipes, support tickets, and unauthenticated log-ins
        event_types = ["withdrawal", "failed_swipe", "failed_swipe", "support_ticket", "login"]
        device_authenticated = random.choice([1, 0, 0, 0]) # High probability of unauthenticated devices
        amount = round(random.uniform(5.0, 300.0), 2) if random.random() > 0.7 else 0.0
    else:
        # Healthy users mostly make clean deposits and withdrawals on authenticated devices
        event_types = ["deposit", "deposit", "withdrawal", "login"]
        device_authenticated = random.choice([1, 1, 1, 0])
        amount = round(random.uniform(50.0, 1500.0), 2) if random.random() > 0.3 else 0.0
        
    event = random.choice(event_types)
    
    # Rare cross-contamination: Occasionally give a healthy user a friction event or vice versa
    if random.random() < 0.05:
        event = random.choice(["failed_swipe", "support_ticket"])

    return {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "event_type": event,
        "amount": amount,
        "device_authenticated": device_authenticated
    }

def start_stream_simulation(duration_seconds: int = 15, delay: float = 0.1):
    config = load_config()
    raw_path = config["paths"]["raw_data_path"]
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    print(f"[Data Stream] Booting event engine targeting: {raw_path}")
    
    # Create an explicit pool of 30 users, and tag 25% of them as inherently high-friction
    customer_pool = []
    for _ in range(30):
        uid = f"USR_{random.randint(10000, 99999)}"
        is_high_risk = random.random() < 0.25  # ~25% class imbalance split
        customer_pool.append({"id": uid, "is_high_risk": is_high_risk})
    
    file_exists = os.path.exists(raw_path)
    
    with open(raw_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["timestamp", "customer_id", "event_type", "amount", "device_authenticated"])
        
        start_time = time.time()
        count = 0
        while time.time() - start_time < duration_seconds:
            # Pick a customer and generate profile-specific behavioral events
            chosen_customer = random.choice(customer_pool)
            tx = generate_mock_transaction(chosen_customer["id"], chosen_customer["is_high_risk"])
            
            writer.writerow([tx["timestamp"], tx["customer_id"], tx["event_type"], tx["amount"], tx["device_authenticated"]])
            count += 1
            time.sleep(delay)
            
    print(f"[Data Stream] Simulation complete. Broadcasted {count} live events successfully.")

if __name__ == "__main__":
    start_stream_simulation()