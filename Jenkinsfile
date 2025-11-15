pipeline {
    agent any
    stages {
        stage('Select Best Model') {
            steps {
                sh 'python src/auto_deploy_best.py'
            }
        }
    }
}