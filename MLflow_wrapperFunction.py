import papermill as pm
import mlflow

def run_notebook_with_mlflow(notebook_path, output_path, params):
    with mlflow.start_run() as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_path,
            parameters=params
        )
        
        mlflow.log_artifact(output_path)

        print(f"MLflow run logged: {mlflow.get_artifact_uri()}")
