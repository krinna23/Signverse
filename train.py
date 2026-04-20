"""
train.py
========
Trains three classifiers on hand-landmark data and prints a comparison report.

Models
------
  1  MLP          (Multi-Layer Perceptron)
  2  Random Forest
  3  KNN          (K-Nearest Neighbours) — historically best

Usage
-----
    python train.py

Output
------
  model_mlp.p   model_rf.p   model_knn.p
"""

import pickle
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_FILE = './data.pickle'
try:
    data_dict = pickle.load(open(DATA_FILE, 'rb'))
except FileNotFoundError:
    print(f"[ERROR] {DATA_FILE} not found. Run preprocess.py first.")
    sys.exit(1)

data       = np.asarray(data_dict['data'],   dtype=np.float64)  # float64 avoids casting errors
raw_labels = np.asarray(data_dict['labels'])                     # string labels e.g. 'A','10'

# Encode string labels → integers (sklearn metrics work safely on integers)
le = LabelEncoder()
labels = le.fit_transform(raw_labels)   # e.g. '0'->0, '10'->1, 'A'->11, ...

print(f"{'─'*55}")
print(f"  Loaded  : {len(data)} samples, {data.shape[1]} features each")
print(f"  Classes : {sorted(np.unique(raw_labels).tolist())}")
print(f"{'─'*55}\n")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)
print(f"  Train: {len(X_train)}   Test: {len(X_test)}\n")

# ── Model definitions ──────────────────────────────────────────────────────────
MODELS = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=700,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean',
        n_jobs=-1
    ),
}

SAVE_PATHS = {
    "MLP":           "model_mlp.p",
    "Random Forest": "model_rf.p",
    "KNN":           "model_knn.p",
}

# ── Training & evaluation loop ─────────────────────────────────────────────────
print("--- Model Comparison Report ---\n")
results    = {}
all_preds  = {}

for name, model in MODELS.items():
    print(f"  Training {name} ...", end='', flush=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred) * 100
    results[name]   = acc
    all_preds[name] = y_pred
    print(f"  done   →  Accuracy: {acc:.2f}%")

# ── Detailed report for best model ────────────────────────────────────────────
best_name  = max(results, key=results.get)
y_pred_best = all_preds[best_name]

# Decode integer labels back to original class names for the report
y_test_names  = le.inverse_transform(y_test)
y_pred_names  = le.inverse_transform(y_pred_best)
target_names  = [str(c) for c in le.classes_]

print(f"\n--- Detailed per-class report ({best_name} — best model) ---\n")
print(classification_report(y_test_names, y_pred_names,
                             target_names=None, zero_division=0))

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'─'*42}")
print(f"  {'Model':<18} {'Accuracy':>10}")
print(f"{'─'*42}")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    marker = "  ← best" if name == best_name else ""
    print(f"  {name:<18} {acc:>9.2f}%{marker}")
print(f"{'─'*42}\n")

# ── Save all models + encoder ──────────────────────────────────────────────────
for name, model in MODELS.items():
    path = SAVE_PATHS[name]
    with open(path, 'wb') as f:
        # Store model AND the label encoder so camera_test can decode predictions
        pickle.dump({'model': model, 'label_encoder': le}, f)
    print(f"  Saved  {name:<18} → {path}")

print("\nAll models saved successfully.")
