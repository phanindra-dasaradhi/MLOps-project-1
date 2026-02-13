pipeline{
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                script {
                echo 'Cloning repository...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'phanindra-dasaradhi', url: 'https://github.com/phanindra-dasaradhi/MLOps-project-1.git']])
            }
        }
    }
}