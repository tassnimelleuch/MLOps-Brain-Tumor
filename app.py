from flask import Flask, request, jsonify
import numpy as np
import os
import json
import logging
from tensorflow.keras.models import load_model
import pickle
import sys

# Add src to path for imports
sys.path.append('/app/src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
model_info = None

def load_model_from_export():
    """Load the exported best model"""
    global model, model_info
    
    try:
        # Load model info
        with open('/app/best_model/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Loading model: {model_info['name']} (Accuracy: {model_info['accuracy']:.4f})")
        
        # Load the actual model based on type
        if model_info['type'] == 'cnn':
            model = load_model('/app/best_model/model.h5')
        else:
            with open('/app/best_model/model.pkl', 'rb') as f:
                model = pickle.load(f)
        
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model at startup
if not load_model_from_export():
    logger.error("Failed to initialize model - exiting")
    exit(1)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": model_info['name'],
        "accuracy": model_info['accuracy'],
        "type": model_info['type']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = np.array(data['image'])
        
        # Validate input shape
        if model_info['type'] == 'cnn' and image_data.shape != (150, 150, 3):
            return jsonify({"error": f"Expected shape (150, 150, 3), got {image_data.shape}"}), 400
        
        # Make prediction based on model type
        if model_info['type'] == 'cnn':
            prediction = model.predict(image_data.reshape(1, 150, 150, 3))
            result = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
        else:
            # For traditional ML models, use feature extraction
            from feature_extraction import extract_features
            features = extract_features(image_data.reshape(1, 150, 150, 3))
            prediction = model.predict(features)
            result = int(prediction[0])
            confidence = float(model.predict_proba(features).max())
        
        return jsonify({
            "prediction": result,  # 0 = no tumor, 1 = tumor
            "confidence": confidence,
            "model": model_info['name'],
            "accuracy": model_info['accuracy'],
            "type": model_info['type']
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info_endpoint():
    """Get model information"""
    return jsonify(model_info)

if __name__ == "__main__":
    logger.info("🚀 Starting Brain Tumor Classification API...")
    app.run(host='0.0.0.0', port=8000, debug=False)