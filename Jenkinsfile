pipeline {
    agent any
    
    stages {
        stage('Select Best Model') {
            steps {
                sh '''
                echo "🔍 Selecting best model from MLflow..."
                pip3 install mlflow
                python3 src/auto_deploy_best.py
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh '''
                echo "🐳 Building Docker image..."
                docker build -t brain-tumor-model:latest .
                '''
            }
        }
        
        stage('Test Docker Image') {
            steps {
                sh '''
                echo "🧪 Testing Docker image..."
                docker run -d --name test-model -p 8000:8000 brain-tumor-model:latest
                sleep 10
                curl -f http://localhost:8000/health || exit 1
                docker stop test-model
                docker rm test-model
                '''
            }
        }
        
        stage('Push to Registry') {
            steps {
                sh '''
                echo "📦 Pushing to Docker Registry..."
                docker tag brain-tumor-model:latest your-registry/brain-tumor-model:latest
                docker push your-registry/brain-tumor-model:latest
                '''
            }
        }
    }
    
    post {
        success {
            echo "🎉 SUCCESS: Best model deployed to Docker!"
            archiveArtifacts artifacts: 'production_model.pkl'
        }
        failure {
            echo "❌ FAILED: Pipeline failed!"
        }
    }
}