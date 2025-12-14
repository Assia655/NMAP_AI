import json
import random

# === CONFIG ===
INPUT_FILE = "nmap_dataset.json"   # your corrected full dataset
TRAIN_FILE = "train.json"
TEST_FILE = "test.json"
TRAIN_RATIO = 0.8
SEED = 42                  # for reproducibility

# === LOAD DATA ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# === SHUFFLE ===
random.seed(SEED)
random.shuffle(data)

# === SPLIT ===
split_idx = int(len(data) * TRAIN_RATIO)
train_data = data[:split_idx]
test_data = data[split_idx:]

# === SAVE ===
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4)

with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4)

print(f"Train samples: {len(train_data)} → {TRAIN_FILE}")
print(f"Test samples:  {len(test_data)} → {TEST_FILE}")
print("\n[✔] Shuffle + train/test split completed.")
