pipeline {
    agent {
        docker {
            image 'python:3.9-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock'  
        }
    }
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials')
        DOCKER_IMAGE = 'tasnimelleuchenis/brain-tumor-classifier'
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh '''
                    pip install -r requirements.txt
                    pip install mlflow joblib
                '''
            }
        }
        
        stage('Get Best Model') {
            steps {
                sh 'python3 src/select_best_model.py'
            }
        }
        
        stage('Build & Push Docker') {
            steps {
                sh '''
                    docker build -t $DOCKER_IMAGE .
                    docker push $DOCKER_IMAGE
                '''
            }
        }
    }
}