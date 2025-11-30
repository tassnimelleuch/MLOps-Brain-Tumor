import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os

def get_best_model():
    # USE localhost since MLflow is on same machine as Jenkins
    mlflow.set_tracking_uri("http://localhost:5000")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name("brain-tumor-classification")
    
    if not experiment:
        print("❌ No experiment found")
        return None
    
    # Get best run by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if not runs:
        print("❌ No runs found")
        return None
    
    best_run = runs[0]
    print(f"🏆 Best Run ID: {best_run.info.run_id}")
    print(f"📊 Best Accuracy: {best_run.data.metrics.get('accuracy', 'N/A')}")
    
    # Download model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"📦 Downloading model from: {model_uri}")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Save for Docker
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/best_model.pkl")
        print("✅ Model saved to models/best_model.pkl")
        
        return best_run.info.run_id
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

if __name__ == "__main__":
    get_best_model()