from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
import os

def load_images():
    """Load images directly in this file"""
    data_path = "data/brain-tumor-data"
    X, y = [], []
    
    for class_name in ['no', 'yes']:
        class_path = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                X.append(img)
                y.append(0 if class_name == 'no' else 1)
    
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def extract_features(images):
    """Extract simple features from images"""
    features_list = []
    
    for image in images:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract simple features
        feature_vector = []
        
        # 1. Basic statistical features
        feature_vector.append(np.mean(gray))
        feature_vector.append(np.std(gray))
        feature_vector.append(np.median(gray))
        
        # 2. Simple histogram (first 5 bins)
        hist = cv2.calcHist([gray], [0], None, [5], [0, 256])
        feature_vector.extend(hist.flatten())
        
        features_list.append(feature_vector)
    
    features_array = np.array(features_list)
    print(f"Extracted features shape: {features_array.shape}")
    return features_array

def train_svm():
    """Train SVM model with MLflow tracking"""
    print("Loading data...")
    X, y = load_images()
    
    print("Extracting features...")
    X_features = extract_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    with mlflow.start_run(run_name="SVM_Training"):
        # Log parameters
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("features_shape", X_train.shape[1])
        
        # Create and train SVM
        print("Training SVM...")
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(svm_model, "svm_model")
        
        print(f"🎯 SVM Test Accuracy: {accuracy:.4f}")
        print("✅ SVM training completed with MLflow tracking!")
        
        return svm_model

# THIS MAKES IT RUN WHEN YOU CALL THE FILE
if __name__ == "__main__":
    train_svm()