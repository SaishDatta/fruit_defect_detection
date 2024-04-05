pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the source code from Git 

                git 'https://github.com/SaishDatta/fruit_defect_detection.git'
            }
        }

        stage('Build') {
            steps {
                // Build Docker image
                script {
                    docker.build("saishd/fruit_defect_detection:${env.BUILD_ID}")
                }
            }
        }

        // stage('Test') {
        //     steps {
        //         // Run tests if needed
        //         // Example: sh 'npm test'
        //     }
        }

        stage('Push') {
            steps {
                // Push the Docker image to a Docker registry
                script  {
            docker.withRegistry('https://hub.docker.com/repository/docker/saishd/fruit_defect_detection', 'your-docker-registry-credentials-id') {
                docker.image("saishd/fruit_defect_detection:${env.BUILD_ID}").push()
                    }
                }
            }
        }
    }

    post {
        always {
            // Clean up any temporary resources
            cleanWs()
        }
    }
}
