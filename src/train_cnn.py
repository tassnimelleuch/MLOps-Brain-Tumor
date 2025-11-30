import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split


MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("brain-tumor-classification") 

print(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"📊 Experiment: brain-tumor-classification")

def load_images():
    """Load images directly in this file"""
    data_path = "../data/brain-tumor-data"
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
    
    X = np.array(X) / 255.0 
    y = np.array(y)
    print(f"✅ Loaded {len(X)} images")
    print(f"   Class distribution: {np.bincount(y)}")
    return X, y

def create_cnn_model():
    """Create a CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # 2 classes: no=0, yes=1
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn():
    """Train CNN model with MLflow tracking"""
    print("\n" + "="*50)
    print("🧠 Training CNN Model")
    print("="*50 + "\n")
    
    print("1️⃣ Loading data...")
    X, y = load_images()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}\n")

    with mlflow.start_run(run_name="CNN"):
        
        print("2️⃣ Logging parameters to MLflow...")
        # Log parameters
        mlflow.log_param("model_name", "CNN")  
        mlflow.log_param("model_type", "Convolutional Neural Network")
        mlflow.log_param("input_shape", "150x150x3")
        mlflow.log_param("num_classes", 2)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("dropout", 0.5)
        
        print("3️⃣ Creating CNN model...")
        model = create_cnn_model()
        print("   ✅ Model created!\n")
        
        print("4️⃣ Training model...")
        print("-" * 50)
        
        
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        print("-" * 50)
        print("   ✅ Training completed!\n")
        
        print("5️⃣ Evaluating model...")
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for more metrics
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   🎯 Test Accuracy:  {test_accuracy:.4f}")
        print(f"   🎯 Test Loss:      {test_loss:.4f}")
        print(f"   🎯 Precision:      {precision:.4f}")
        print(f"   🎯 Recall:         {recall:.4f}")
        print(f"   🎯 F1 Score:       {f1:.4f}\n")
        
        print("6️⃣ Logging metrics to MLflow...")
        # Log metrics - IMPORTANT: "accuracy" est utilisé par auto_deploy_best.py
        mlflow.log_metric("accuracy", test_accuracy)  # ← Clé pour la comparaison
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log training history
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        print("7️⃣ Saving model to MLflow...")
        # Log model
        mlflow.tensorflow.log_model(model, "model")  # ← Sauvegarde dans mlruns/
        
        run_id = mlflow.active_run().info.run_id
        print(f"   ✅ Model saved with Run ID: {run_id}\n")
        
        print("="*50)
        print("✅ CNN Training Completed Successfully!")
        print("="*50)
        print(f"📊 View results: http://localhost:5000")
        print(f"🆔 Run ID: {run_id}")
        print("="*50 + "\n")
        
        return model, history

if __name__ == "__main__":
    train_cnn()