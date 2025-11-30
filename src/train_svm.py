from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("brain-tumor-classification")

print(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"📊 Experiment: brain-tumor-classification")
# ============================================

def load_images():
    """Load images directly in this file"""
    data_path = "../data/brain-tumor-data"
    X, y = [], []
    
    for class_name in ['no', 'yes']:
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: Path not found: {class_path}")
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                X.append(img)
                y.append(0 if class_name == 'no' else 1)
    
    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} images")
    print(f"   Class distribution: {np.bincount(y)}")
    return X, y

def extract_features(images):
    """Extract simple features from images for SVM"""
    features_list = []
    
    print("   Extracting features from images...")
    for i, image in enumerate(images):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i + 1}/{len(images)} images processed")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract features
        feature_vector = []
        
        # 1. Basic statistical features
        feature_vector.append(np.mean(gray))
        feature_vector.append(np.std(gray))
        feature_vector.append(np.median(gray))
        feature_vector.append(np.min(gray))
        feature_vector.append(np.max(gray))
        
        # 2. Histogram features (10 bins)
        hist = cv2.calcHist([gray], [0], None, [10], [0, 256])
        feature_vector.extend(hist.flatten())
        
        # 3. Edge detection features
        edges = cv2.Canny(gray, 100, 200)
        feature_vector.append(np.mean(edges))
        feature_vector.append(np.std(edges))
        
        features_list.append(feature_vector)
    
    features_array = np.array(features_list)
    print(f"   ✅ Extracted features shape: {features_array.shape}")
    return features_array

def train_svm():
    """Train SVM model with MLflow tracking"""
    print("\n" + "="*50)
    print("🤖 Training SVM Model")
    print("="*50 + "\n")
    
    print("1️⃣ Loading data...")
    X, y = load_images()
    
    print("\n2️⃣ Extracting features...")
    X_features = extract_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}\n")
    
 
    with mlflow.start_run(run_name="SVM"):
        
        print("3️⃣ Logging parameters to MLflow...")
        # Log parameters
        mlflow.log_param("model_name", "SVM")  # ← IMPORTANT pour auto_deploy_best.py
        mlflow.log_param("model_type", "Support Vector Machine")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("features_shape", X_train.shape[1])
        mlflow.log_param("C", 1.0)
        mlflow.log_param("gamma", "scale")
        
        print("4️⃣ Creating and training SVM model...")
        print("-" * 50)
        
        # Create and train SVM
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True  # Pour avoir les probabilités de prédiction
        )
        svm_model.fit(X_train, y_train)
        
        print("-" * 50)
        print("   ✅ Training completed!\n")
        
        print("5️⃣ Evaluating model...")
        # Make predictions
        y_pred = svm_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   🎯 Test Accuracy:  {accuracy:.4f}")
        print(f"   🎯 Precision:      {precision:.4f}")
        print(f"   🎯 Recall:         {recall:.4f}")
        print(f"   🎯 F1 Score:       {f1:.4f}\n")
        
        print("6️⃣ Logging metrics to MLflow...")
        # Log metrics - IMPORTANT: "accuracy" est utilisé par auto_deploy_best.py
        mlflow.log_metric("accuracy", accuracy)  # ← Clé pour la comparaison
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print("7️⃣ Saving model to MLflow...")
        # Log model
        mlflow.sklearn.log_model(svm_model, "model")  # ← Sauvegarde dans mlruns/
        
        run_id = mlflow.active_run().info.run_id
        print(f"   ✅ Model saved with Run ID: {run_id}\n")
        
        print("="*50)
        print("✅ SVM Training Completed Successfully!")
        print("="*50)
        print(f"📊 View results: http://localhost:5000")
        print(f"🆔 Run ID: {run_id}")
        print("="*50 + "\n")
        
        return svm_model

if __name__ == "__main__":
    train_svm()