import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Load trained classifiers + label encoders ──────────────────────────────────
MODEL_FILES = {
    'MLP': 'model_mlp.p',
    'RF':  'model_rf.p',
    'KNN': 'model_knn.p',
}

models = {}  # model_type -> (classifier, label_encoder)

for key, path in MODEL_FILES.items():
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                pkg = pickle.load(f)
                clf = pkg['model']
                le = pkg.get('label_encoder', None)
                models[key] = (clf, le)
                logger.info(f"Loaded {key} from {path}")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
    else:
        logger.warning(f"{path} not found")

FEATURES_PER_HAND = 63
ZERO_HAND = [0.0] * FEATURES_PER_HAND


def decode_prediction(raw_pred, le):
    if le is not None:
        try:
            return str(le.inverse_transform([int(raw_pred)])[0])
        except Exception:
            return str(raw_pred)
    return str(raw_pred)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    landmarks = data.get('landmarks')  # Expected: {'Left': [...], 'Right': [...]}
    model_type = data.get('model_type', 'RF')

    if not landmarks:
        return jsonify({'error': 'No landmarks provided'}), 400

    if model_type not in models:
        model_type = 'RF' if 'RF' in models else list(models.keys())[0]

    clf, le = models[model_type]

    # Preprocess landmarks (must match training feature extraction)
    left_hand = landmarks.get('Left', ZERO_HAND)
    right_hand = landmarks.get('Right', ZERO_HAND)

    # Validate feature lengths
    if not isinstance(left_hand, list) or len(left_hand) != FEATURES_PER_HAND:
        left_hand = ZERO_HAND
    if not isinstance(right_hand, list) or len(right_hand) != FEATURES_PER_HAND:
        right_hand = ZERO_HAND

    feature_vector = np.array(left_hand + right_hand, dtype=np.float64).reshape(1, -1)

    # Predict
    try:
        raw_pred = clf.predict(feature_vector)[0]
        prediction = decode_prediction(raw_pred, le)

        # Get confidence if supported
        confidence = 100.0
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(feature_vector)[0]
            confidence = float(np.max(proba)) * 100

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'model': model_type
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'features_per_hand': FEATURES_PER_HAND,
        'total_features': FEATURES_PER_HAND * 2,
    })


if __name__ == '__main__':
    logger.info(f"Models loaded: {list(models.keys())}")
    logger.info(f"Feature vector size: {FEATURES_PER_HAND * 2}")
    app.run(port=5000, debug=True)
