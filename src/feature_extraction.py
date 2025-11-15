import cv2
import numpy as np
from skimage import feature

def extract_features(images):
    """Extract features from images for traditional ML models"""
    features_list = []
    
    for image in images:
        # Convert to uint8 and handle color conversion
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract simple features (avoid complex ones for now)
        feature_vector = []
        
        # 1. Basic statistical features
        feature_vector.append(np.mean(gray))
        feature_vector.append(np.std(gray))
        feature_vector.append(np.median(gray))
        
        # 2. Simple histogram (first 10 bins)
        hist = cv2.calcHist([gray], [0], None, [10], [0, 256])
        feature_vector.extend(hist.flatten())
        
        features_list.append(feature_vector)
    
    print(f"Extracted features shape: {np.array(features_list).shape}")
    return np.array(features_list)