from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load the model
with open('production_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        image_data = base64.b64decode(data['image'])
        
        # Preprocess image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        result = "Tumor" if prediction[0] > 0.5 else "No Tumor"
        confidence = float(prediction[0])
        
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "model_type": "CNN"  # You can get this from MLflow
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)