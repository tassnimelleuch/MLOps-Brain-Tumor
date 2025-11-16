import mlflow
import pandas as pd
import os
import pickle

def auto_deploy_best_model():
    """AUTOMATICALLY find and deploy the best fucking model"""
    
    # SET THE CORRECT PORT - 5001 instead of 5000!
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Get all experiments
    experiment = mlflow.get_experiment_by_name("Brain_Tumor_Classification")
    
    if experiment is None:
        print("❌ Experiment 'Brain_Tumor_Classification' not found!")
        print("💡 Check if MLflow server is running on port 5001")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Find the run with highest accuracy
    best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
    
    model_name = best_run['tags.mlflow.runName']
    accuracy = best_run['metrics.test_accuracy']
    run_id = best_run['run_id']
    
    print(f"🔥 BEST MODEL FOUND: {model_name}")
    print(f"🎯 ACCURACY: {accuracy:.4f}")
    
    # Load the best fucking model
    if "CNN" in model_name:
        model_uri = f"runs:/{run_id}/cnn_model"
        model = mlflow.tensorflow.load_model(model_uri)
    elif "XGBoost" in model_name:
        model_uri = f"runs:/{run_id}/xgboost_model" 
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model_uri = f"runs:/{run_id}/svm_model"
        model = mlflow.sklearn.load_model(model_uri)
    
    with open('production_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"🚀 AUTOMATICALLY DEPLOYED: {model_name}")
    return model

if __name__ == "__main__":
    auto_deploy_best_model()