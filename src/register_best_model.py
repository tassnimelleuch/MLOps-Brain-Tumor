import mlflow
import pandas as pd

def register_best_model():
    """Register the best model in MLflow Model Registry"""
    
    # Find best model from experiments
    experiment = mlflow.get_experiment_by_name("Brain_Tumor_Classification")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
    
    model_name = best_run['tags.mlflow.runName']
    model_uri = f"runs:/{best_run['run_id']}/model"
    
    print(f"🏆 Best Model: {model_name}")
    print(f"🎯 Accuracy: {best_run['metrics.test_accuracy']:.4f}")
    
    # Register model
    mlflow.register_model(
        model_uri=model_uri,
        name="brain-tumor-classifier"
    )
    
    print("✅ Model registered in MLflow Model Registry!")
    
    return best_run

if __name__ == "__main__":
    register_best_model()