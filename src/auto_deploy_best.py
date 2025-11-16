import mlflow
import os
import pickle

def auto_deploy_best_model():
    # Set your MLflow tracking URI (CHANGE THIS TO YOUR SERVER)
    mlflow.set_tracking_uri("http://localhost:5000")  # or your actual MLflow server
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name("Brain_Tumor_Classification")
    
    if experiment is None:
        print("❌ Experiment 'Brain_Tumor_Classification' not found!")
        print("💡 Make sure MLflow tracking URI is correct and experiment exists.")
        return None
    
    # Find best model
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("❌ No runs found in experiment!")
        return None
    
    best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
    model_name = best_run['tags.mlflow.runName']
    accuracy = best_run['metrics.test_accuracy']
    
    print(f"🔥 BEST MODEL: {model_name} (Accuracy: {accuracy:.4f})")
    
    # Save dummy model for now (replace with actual model loading)
    dummy_model = {"model_name": model_name, "accuracy": accuracy}
    with open('production_model.pkl', 'wb') as f:
        pickle.dump(dummy_model, f)
    
    print("✅ Best model info saved!")
    return dummy_model

if __name__ == "__main__":
    auto_deploy_best_model()