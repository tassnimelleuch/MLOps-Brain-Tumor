pipeline {
    agent any
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials')
        DOCKER_IMAGE = 'tasnimelleuchenis/brain-tumor-classifier'
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                git 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git'
            }
        }
        
        stage('Get Best Model') {
            steps {
                sh 'python src/select_best_model.py'
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