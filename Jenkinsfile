pipeline {
    agent any
    stages {
        stage('Select Best Model') {
            steps {
                sh 'python3 src/auto_deploy_best.py'
            }
        }
    }
}