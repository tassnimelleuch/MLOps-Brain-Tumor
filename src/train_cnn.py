import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

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
    
    X = np.array(X) / 255.0  # Normalize
    y = np.array(y)
    print(f"Loaded {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
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
    print("Loading data...")
    X, y = load_images()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    with mlflow.start_run(run_name="CNN_Training"):
        # Log parameters
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("input_shape", X_train.shape[1:])
        mlflow.log_param("num_classes", 2)
        mlflow.log_param("epochs", 5)  # Reduced for testing
        mlflow.log_param("batch_size", 32)
        
        # Create model
        model = create_cnn_model()
        print("Model created!")
        
        # Train model
        print("Starting training...")
        history = model.fit(X_train, y_train,
                          epochs=5,
                          batch_size=32,
                          validation_data=(X_test, y_test),
                          verbose=1)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 CNN Test Accuracy: {test_accuracy:.4f}")
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        
        # Log model
        mlflow.tensorflow.log_model(model, "cnn_model")
        
        print("✅ CNN training completed with MLflow tracking!")
        
        return model, history

# THIS MAKES IT RUN WHEN YOU CALL THE FILE
if __name__ == "__main__":
    train_cnn()