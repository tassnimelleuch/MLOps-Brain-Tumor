pipeline {
    agent any
    
    stages {
        stage('Auto Deploy Best Model') {
            steps {
                git branch: 'main', 
                url: 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git',
                credentialsId: 'github-token'
                
                sh '''
                pip3 install -r requirements.txt
                python3 src/auto_deploy_best.py
                '''
            }
        }
    }
    
    post {
        success {
            archiveArtifacts artifacts: 'production_model.pkl'
            echo " SUCCESS: Best model selected and saved!"
        }
    }
}