pipeline {
    agent any
    
    stages {
        stage('Checkout Code & Model Info') {
            steps {
                git branch: 'main', url: 'https://github.com/your-username/MLOps-Brain-Tumor.git'
            }
        }
        
        stage('Get Best Model from Registry') {
            steps {
                bat '''
                call venv\\Scripts\\activate && 
                python src/get_model_from_registry.py
                '''
            }
        }
        
        stage('Test Model') {
            steps {
                bat '''
                call venv\\Scripts\\activate && 
                python src/test_model_validation.py
                '''
            }
        }
        
        stage('Deploy Model') {
            steps {
                bat '''
                call venv\\Scripts\\activate && 
                python src/deploy_model.py
                '''
            }
        }
    }
}