"""
test_accuracy.py
================
Loads the three saved ISL models and evaluates their accuracy on the data.
This script ensures that the saved models are performing as expected.

Usage
-----
    python test_accuracy.py
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_FILE = './data.pickle'
MODEL_FILES = {
    "MLP":           "model_mlp.p",
    "Random Forest": "model_rf.p",
    "KNN":           "model_knn.p",
}

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found.")
        return None, None
    
    with open(DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = np.asarray(data_dict['data'], dtype=np.float64)
    labels = np.asarray(data_dict['labels'])
    return data, labels

def evaluate_models():
    print(f"{'='*60}")
    print(f"  ISL Model Accuracy Evaluation")
    print(f"{'='*60}\n")

    data, raw_labels = load_data()
    if data is None: return

    # We use the same split as train.py to verify generalization
    # If you want to check the entire dataset, you can omit the split
    X_train, X_test, y_train, y_test = train_test_split(
        data, raw_labels,
        test_size=0.2,
        shuffle=True,
        stratify=raw_labels,
        random_state=42
    )

    print(f"  Total samples : {len(data)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"  Classes       : {len(np.unique(raw_labels))}\n")

    results = {}

    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name} model file ({path}) not found.")
            continue

        print(f"  Evaluating {name:<15} ...", end='', flush=True)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            le = model_data['label_encoder']

        y_test_encoded = le.transform(y_test)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test_encoded, y_pred) * 100
        results[name] = acc
        print(f" done -> Accuracy: {acc:.2f}%")

    # ── Summary Report ────────────────────────────────────────────────────────
    if not results:
        print("\nNo models were evaluated.")
        return

    print(f"\n{'-'*45}")
    print(f"  {'Model':<20} {'Accuracy':>10}")
    print(f"{'-'*45}")
    # Sort by accuracy descending
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {acc:>9.2f}%")
    print(f"{'-'*45}\n")

    # Show detailed report for the best performing model
    best_name = max(results, key=results.get)
    print(f"Detailed Report for Best Model: {best_name}")
    
    with open(MODEL_FILES[best_name], 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        le = model_data['label_encoder']
    
    y_test_encoded = le.transform(y_test)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_, zero_division=0))

if __name__ == "__main__":
    evaluate_models()
