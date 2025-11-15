pipeline {
    agent any
    
    triggers {
        githubPush()  // Webhook trigger
    }
    
    stages {
        stage('Checkout from GitHub') {
            steps {
                echo "🚀 Testing GitHub connection..."
                git branch: 'main', 
                url: 'https://github.com/tassnimelleuch/MLOps-Brain-Tumor.git',
                credentialsId: 'github-token'
                
                // List files to confirm checkout worked
                bat 'dir'
            }
        }
        
        stage('Verify Project Structure') {
            steps {
                echo "📁 Checking project structure..."
                bat '''
                echo "=== Project Files ==="
                dir src\\
                echo "=== Requirements ==="
                type requirements.txt
                '''
            }
        }
    }
    
    post {
        always {
            echo "✅ GitHub connection test completed!"
        }
        success {
            echo "🎉 SUCCESS: GitHub webhook is working!"
            // Archive the successful build info
            archiveArtifacts artifacts: '**/*.txt,**/*.py', excludes: 'venv/**'
        }
        failure {
            echo "❌ FAILED: Check GitHub credentials or network"
        }
    }
}