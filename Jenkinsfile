pipeline {
    agent any


    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage('Clone Repository') {
            steps {
                echo 'Cloning repository...'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[
                        credentialsId: 'phanindra-dasaradhi',
                        url: 'https://github.com/phanindra-dasaradhi/MLOps-project-1.git'
                    ]]
                ])
            }
        }

        stage('Set Up Python Environment') {
            steps {
                echo 'Setting up Python environment...'
                sh '''
                    python -m venv ${VENV_DIR}
                    source ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                '''
            }
        }
    }
}
