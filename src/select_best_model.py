import mlflow
import joblib
import os

mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Get best model from MLflow
best_run = mlflow.search_runs(
    experiment_names=["brain-tumor-classification"],
    order_by=["metrics.accuracy DESC"]
).iloc[0]

model = mlflow.pyfunc.load_model(f"runs:/{best_run.run_id}/model")

# Save for Docker
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")