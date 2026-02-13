pipeline {
    agent any

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
    }
}
