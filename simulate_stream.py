"""
simulate_stream.py
------------------
Continuously generates synthetic customer events and appends them to
data/live_feed.csv so the Live Monitor dashboard page can show real-time updates.

Run in a separate terminal:
    python simulate_stream.py

Stop with Ctrl+C.
"""

import os
import time
import signal
import sys
import numpy as np
import pandas as pd
import joblib

LIVE_FEED_PATH = "data/live_feed.csv"
INTERVAL_SEC   = 2          # new event every 2 seconds
MAX_ROWS       = 500        # keep rolling window of last 500 events

CONTRACT_TYPES    = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS   = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]

COLUMNS = [
    "timestamp", "customer_id",
    "tenure", "monthly_charges", "total_charges",
    "num_services", "support_calls",
    "contract_type", "payment_method", "internet_service",
    "churn_probability", "predicted_churn",
]


def generate_event(rng: np.random.Generator, customer_id: int) -> dict:
    tenure          = int(rng.integers(1, 73))
    monthly_charges = round(float(rng.uniform(20, 120)), 2)
    total_charges   = round(monthly_charges * tenure * float(rng.uniform(0.85, 1.05)), 2)
    num_services    = int(rng.integers(1, 8))
    support_calls   = int(rng.integers(0, 11))
    contract_type   = rng.choice(CONTRACT_TYPES,    p=[0.55, 0.25, 0.20])
    payment_method  = rng.choice(PAYMENT_METHODS)
    internet_service = rng.choice(INTERNET_SERVICES, p=[0.35, 0.45, 0.20])

    return {
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "num_services":     num_services,
        "support_calls":    support_calls,
        "contract_type":    contract_type,
        "payment_method":   payment_method,
        "internet_service": internet_service,
        "customer_id":      customer_id,
    }


def load_model():
    if not os.path.exists("model/churn_model.joblib"):
        print("[ERROR] Model not found. Run python train_model.py first.")
        sys.exit(1)
    model        = joblib.load("model/churn_model.joblib")
    preprocessor = joblib.load("model/preprocessor.joblib")
    return model, preprocessor


def init_feed():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LIVE_FEED_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(LIVE_FEED_PATH, index=False)
        print(f"[*] Created {LIVE_FEED_PATH}")
    else:
        print(f"[*] Appending to existing {LIVE_FEED_PATH}")


def handle_exit(sig, frame):
    print("\n[*] Stream simulator stopped.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print("[*] Loading model ...")
    model, preprocessor = load_model()

    init_feed()

    rng          = np.random.default_rng()
    customer_id  = 10001
    event_count  = 0

    print(f"[*] Streaming new customer events every {INTERVAL_SEC}s  (Ctrl+C to stop)\n")

    while True:
        event = generate_event(rng, customer_id)

        # Predict
        feature_cols = [
            "tenure", "monthly_charges", "total_charges",
            "num_services", "support_calls",
            "contract_type", "payment_method", "internet_service",
        ]
        input_df   = pd.DataFrame([{k: event[k] for k in feature_cols}])
        X_proc     = preprocessor.transform(input_df)
        churn_prob = float(model.predict_proba(X_proc)[0, 1])
        predicted  = 1 if churn_prob >= 0.50 else 0

        row = {
            "timestamp":        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id":      customer_id,
            "tenure":           event["tenure"],
            "monthly_charges":  event["monthly_charges"],
            "total_charges":    event["total_charges"],
            "num_services":     event["num_services"],
            "support_calls":    event["support_calls"],
            "contract_type":    event["contract_type"],
            "payment_method":   event["payment_method"],
            "internet_service": event["internet_service"],
            "churn_probability": round(churn_prob, 4),
            "predicted_churn":  predicted,
        }

        # Append row
        pd.DataFrame([row]).to_csv(
            LIVE_FEED_PATH, mode="a", header=False, index=False
        )

        # Trim to rolling window
        df_feed = pd.read_csv(LIVE_FEED_PATH)
        if len(df_feed) > MAX_ROWS:
            df_feed = df_feed.tail(MAX_ROWS)
            df_feed.to_csv(LIVE_FEED_PATH, index=False)

        event_count += 1
        status = "CHURN" if predicted else "OK   "
        print(f"  [{status}]  Customer {customer_id}  |  "
              f"prob={churn_prob:.2%}  |  "
              f"contract={event['contract_type']:<16}  |  "
              f"charges=${event['monthly_charges']:.2f}")

        customer_id += 1
        time.sleep(INTERVAL_SEC)
