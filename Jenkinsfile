pipeline {
    agent any
    
    triggers {
        githubPush()  
    }
    
    environment {
        MLFLOW_TRACKING_URI = 'file:///./mlruns'
    }
    
    stages {
        stage('Checkout from GitHub') {
            steps {
                echo "🚀 Testing GitHub connection..."
                git branch: 'main', 
                url: 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git',
                credentialsId: 'github-token'
            }
        }
        
        stage('Get Best Model from MLflow') {
            steps {
                echo "🔍 Finding best model from MLflow..."
                bat '''
                call venv\\Scripts\\activate
                python src/auto_deploy_best.py
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker container..."
                bat '''
                docker build -t brain-tumor-model:latest -f Dockerfile .
                '''
            }
        }
        
        stage('Test Docker Container') {
            steps {
                echo "🧪 Testing Docker container..."
                bat '''
                docker run --rm brain-tumor-model:latest python test_model.py
                '''
            }
        }
        
        stage('Push to Registry') {
            steps {
                echo "📦 Pushing to Docker Registry..."
                bat '''
                docker tag brain-tumor-model:latest your-registry/brain-tumor-model:latest
                docker push your-registry/brain-tumor-model:latest
                '''
            }
        }
    }
    
    post {
        always {
            echo "✅ MLOps pipeline completed!"
        }
        success {
            echo "🎉 SUCCESS: Model containerized and ready for deployment!"
        }
    }
}