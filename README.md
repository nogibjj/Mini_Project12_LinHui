# Iris Classification ML Project with MLflow

This project demonstrates the use of a machine learning model to classify the Iris dataset. The model is built using the RandomForestClassifier from the scikit-learn library. Additionally, the project employs MLflow for tracking experiments, including parameters and metrics. The project is structured to facilitate continuous integration and deployment using GitHub Actions.

## Structure

`main.py``: Contains the code for loading data, training the model, making predictions, and tracking the experiment with MLflow.
`requirements.txt``: Lists the Python dependencies required for the project.
`Makefile``: Includes commands for setting up the project environment, running tests, formatting code, and linting.
``.github/workflows/cicd.yml``: Defines the GitHub Actions workflow for continuous integration and deployment.

## MLflow Tracking

MLflow was pivotal for tracking experiments and managing the RandomForestClassifier model, enhancing both organization and reproducibility. Its integration into our CI/CD pipeline streamlined the model training and evaluation process, significantly contributing to efficient MLOps practices.

To view the MLflow tracking UI:

1. Run `mlflow ui` or `mlflow ui --port` followed with the number of port of your choice (e.g. `mlflow ui --port 5001`)
2. Open the localhost URL shown in your terminal