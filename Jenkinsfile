pipeline {
    agent any
    
    stages {
        stage('Auto Deploy Best Model') {
            steps {
                git branch: 'main', 
                url: 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git',
                credentialsId: 'github-token'
                
                bat '''
                python -m venv venv
                call venv\\Scripts\\activate
                pip install -r requirements.txt
                python src/auto_deploy_best.py
                '''
            }
        }
    }
    
    post {
        success {
            archiveArtifacts artifacts: 'production_model.pkl'
            echo "🎉 SUCCESS: Best model selected and saved!"
        }
    }
}