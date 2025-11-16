pipeline {
    agent any
    stages {
        stage('Select Best Model') {
            steps {
                sh '''
                pip3 install -r requirements.txt
                python3 src/auto_deploy_best.py
                '''
            }
        }
    }
}