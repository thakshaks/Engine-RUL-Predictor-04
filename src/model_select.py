import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

db_path = Path("E:/P2/Python/TurbofanEngine/git_FD004/mlflow_artifacts/mlflow.db").resolve().as_uri()
mlflow.set_tracking_uri(f"sqlite:///{db_path.split(':///')[1]}")
# Initialize the MLflow client (pointing to your local sqlite database)
client = MlflowClient()
l1=[]
d1={}
# Specify your experiment name
for i in range(1,6):
    l2=[]
    experiment_name = f"cmapss_turbofan_experiment_{i}"
    try:
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            print(experiment)

            #Find the run with the lowest validation RMSE
            runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_rmse ASC"],
            max_results=1
            )

            if runs:
                best_run = runs[0]
                l2.append(experiment_name)
                l2.append(best_run.data.tags.get("mlflow.runName", "Unnamed Run"))
                l2.append(best_run.data.metrics['val_rmse'])
                l2.append(best_run.info.run_id)
                l1.append(l2)
        # print(f"Best run for {experiment_name}:")
        # print(f"Run ID: {best_run.info.run_id}")
        # print(f"Validation RMSE: {best_run.data.metrics['val_rmse']}")
        # print(f"Validation NASA Score: {best_run.data.metrics['val_nasa_score']}")
        # print(f"Validation MAE: {best_run.data.metrics['val_mae']}")
        # print(f"Validation R2: {best_run.data.metrics['val_r2']}")
        # if 'training_time' in best_run.data.metrics:
        #     print(f"Training Time (seconds): {best_run.data.metrics['training_time']}")
        #print(f"Training Time (seconds): {best_run.data.metrics['training_time']}")
                print("-" * 40)
            else:
                print(f"No runs found for experiment: {experiment_name}")
    except:
        pass


l1_sorted=sorted(l1, key=lambda x:x[2])

for m in l1_sorted:
    print(m)

# best_run_id = l1_sorted[0][3]
# print(best_run_id)
# run_id = best_run_id
# run = client.get_run(run_id)



# model_uri = f"runs:/{best_run_id}/GRU"
# registered_model_name = "CMAPSS_GRU_RUL_Predictor"
    
# # Register the model into the MLflow Model Registry
# model_version = mlflow.register_model(model_uri, registered_model_name)
# print(f"Successfully registered model version {model_version.version} under '{registered_model_name}'")

