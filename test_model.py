import requests
import base64
import cv2
import numpy as np

def test_docker_model():
    # Load a test image
    img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Test prediction
    response = requests.post(
        'http://localhost:8000/predict',
        json={'image': img_base64}
    )
    
    if response.status_code == 200:
        print(f"✅ Docker model test passed: {response.json()}")
    else:
        print(f"❌ Docker model test failed: {response.text}")

if __name__ == "__main__":
    test_docker_model()