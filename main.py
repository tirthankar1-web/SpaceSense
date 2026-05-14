"""
SPACESENSE AI PRO — ML Prediction Engine
=========================================
Reads live distance readings from Firebase (node1/history),
trains an LSTM model, and pushes predictions back to Firebase
so the dashboard can display them in real time.

HOW TO RUN
----------
1. Install dependencies:
       pip install requests numpy tensorflow

   OR if pip gives errors:
       pip install requests numpy tensorflow --break-system-packages

   For Raspberry Pi / ARM:
       pip install requests numpy tflite-runtime

2. Run:
       python spacesense_ai.py

3. It will automatically:
   - Pull all sensor readings from Firebase
   - Train the LSTM model
   - Push predictions to Firebase every cycle
   - Dashboard will show predictions live
"""

import requests
import numpy as np
import time
import sys

# ── Try importing TensorFlow (also works with tflite on Pi) ──────────────────
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("[AI] TensorFlow loaded OK")
except ImportError:
    TF_AVAILABLE = False
    print("[AI] TensorFlow not found — running in SIMPLE AVERAGE mode")
    print("[AI] Install with: pip install tensorflow")

# ── Firebase URLs (must match ESP32 code) ───────────────────────────────────
# ESP32 writes:  PUT  → node1/latest.json   (single clean value)
#                POST → node1/history.json  (all readings, push-keyed)
LATEST_URL  = "https://public-space-detection-default-rtdb.firebaseio.com/node1/latest.json"
HISTORY_URL = "https://public-space-detection-default-rtdb.firebaseio.com/node1/history.json"
PRED_URL    = "https://public-space-detection-default-rtdb.firebaseio.com/prediction.json"

EXPECTED_DISTANCE = 100   # cm — baseline for SEI calculation
MIN_SAMPLES       = 5     # minimum readings before training starts
SEQ_LEN           = 3     # LSTM sequence length
CYCLE_DELAY       = 10    # seconds between prediction cycles


# ── Fetch all history readings from Firebase ─────────────────────────────────
def fetch_history():
    """
    Returns a list of distance floats in chronological order.
    Firebase push-keys are lexicographically chronological so
    sorting by key gives correct time order — no index needed.
    """
    try:
        res = requests.get(HISTORY_URL, timeout=8)
        data = res.json()

        if not data or not isinstance(data, dict):
            # History empty — fall back to latest single value
            res2 = requests.get(LATEST_URL, timeout=8)
            latest = res2.json()
            if latest and "distance" in latest:
                return [float(latest["distance"])]
            return []

        # Sort by push-key (= chronological order)
        sorted_entries = sorted(data.items(), key=lambda x: x[0])

        readings = []
        for key, val in sorted_entries:
            if isinstance(val, dict) and "distance" in val:
                try:
                    d = float(val["distance"])
                    if 2.0 <= d <= 400.0:   # valid sensor range
                        readings.append(round(d, 2))
                except (ValueError, TypeError):
                    pass

        return readings

    except Exception as e:
        print(f"[Firebase] Fetch error: {e}")
        return []


# ── Build LSTM model ──────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(SEQ_LEN, 1), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ── Prepare sequences for LSTM ────────────────────────────────────────────────
def prepare_sequences(series):
    X, y = [], []
    for i in range(len(series) - SEQ_LEN):
        X.append(series[i:i + SEQ_LEN])
        y.append(series[i + SEQ_LEN])
    X = np.array(X, dtype=np.float32).reshape(-1, SEQ_LEN, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


# ── Simple moving average fallback (no TensorFlow needed) ────────────────────
def simple_predict(series):
    weights = np.array([0.2, 0.3, 0.5])   # more weight to recent values
    return float(np.dot(series[-3:], weights))


# ── Push prediction to Firebase ───────────────────────────────────────────────
def send_prediction(pred, sei, loss=None, samples=0):
    payload = {
        "predicted_distance": round(pred, 4),
        "predicted_sei":      round(sei, 4),
        "timestamp":          int(time.time()),
        "samples_used":       samples,
        "model":              "LSTM" if TF_AVAILABLE else "MovingAverage",
    }
    if loss is not None:
        payload["training_loss"] = round(float(loss), 6)

    try:
        res = requests.put(PRED_URL, json=payload, timeout=8)
        print(f"[Firebase] Prediction sent → HTTP {res.status_code}")
        return res.status_code == 200
    except Exception as e:
        print(f"[Firebase] Send error: {e}")
        return False


# ── Print status ──────────────────────────────────────────────────────────────
def print_status(pred, sei):
    print(f"\n{'='*45}")
    print(f"  Predicted Distance : {pred:.2f} cm")
    print(f"  SEI                : {sei:.4f}")
    if sei > 0.5:
        print("  Status             : ENCROACHMENT LIKELY")
    elif sei > 0.2:
        print("  Status             : TEMPORARY OBSTRUCTION")
    else:
        print("  Status             : NORMAL")
    print(f"{'='*45}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*45)
    print("   SPACESENSE AI PRO — ML Prediction Engine")
    print("="*45 + "\n")

    model       = build_model() if TF_AVAILABLE else None
    model_ready = False   # becomes True after first successful training

    cycle = 0

    while True:
        cycle += 1
        print(f"[Cycle {cycle}] Fetching sensor history...")

        series = fetch_history()
        n      = len(series)
        print(f"[Cycle {cycle}] {n} valid readings found")

        if n < MIN_SAMPLES:
            print(f"[Cycle {cycle}] Need at least {MIN_SAMPLES} readings — waiting...")
            time.sleep(CYCLE_DELAY)
            continue

        print(f"[Cycle {cycle}] Last 5 readings: {series[-5:]}")

        loss = None

        if TF_AVAILABLE and model is not None:
            # ── LSTM path ─────────────────────────────────────────
            X, y = prepare_sequences(series)

            if len(X) >= 2:
                early_stop = EarlyStopping(
                    monitor='loss', patience=5, restore_best_weights=True
                )
                epochs = min(30, max(10, n))   # more data → more epochs
                history_obj = model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=max(1, len(X) // 4),
                    verbose=0,
                    callbacks=[early_stop]
                )
                loss        = history_obj.history['loss'][-1]
                model_ready = True
                print(f"[Cycle {cycle}] Training done — loss: {loss:.4f}")

                # Predict next value
                seq  = np.array(series[-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, 1)
                pred = float(model.predict(seq, verbose=0)[0][0])
            else:
                # Not enough sequences yet — use simple fallback
                pred = simple_predict(series)
        else:
            # ── Simple moving average path ────────────────────────
            pred = simple_predict(series)

        # Clamp to realistic sensor range
        pred = max(2.0, min(400.0, pred))

        sei = (EXPECTED_DISTANCE - pred) / EXPECTED_DISTANCE

        print_status(pred, sei)
        send_prediction(pred, sei, loss=loss, samples=n)

        print(f"[Cycle {cycle}] Next prediction in {CYCLE_DELAY}s...\n")
        time.sleep(CYCLE_DELAY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[AI] Stopped by user.")
        sys.exit(0)