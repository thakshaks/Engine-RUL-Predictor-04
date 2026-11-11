import time
import mlflow
import mlflow.pytorch
import mlflow.keras
import mlflow.sklearn
from evaluate import evaluate_predictions
from transfomertrain import train_model

def train_and_log(model, model_name, X_tr, y_tr, X_v, y_v, params, is_dl, is_transformer):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        start = time.perf_counter()

        if is_transformer:
            y_pred = train_model(model, params, X_tr, y_tr, X_v)
        elif is_dl:
            model.fit(X_tr, y_tr,
                      epochs=params.get("epochs", 15),
                      batch_size=params.get("batch_size", 64),
                      verbose=0)
            y_pred = model.predict(X_v).flatten()
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_v)

        elapsed = time.perf_counter() - start
        rmse, mae, r2, nasa = evaluate_predictions(y_v, y_pred)

        if is_transformer:
            mlflow.pytorch.log_model(model, name=model_name, serialization_format="pickle")
        elif is_dl:
            mlflow.keras.log_model(model, name=model_name)
        else: 
            mlflow.sklearn.log_model(
                model,
                name=model_name,
                skops_trusted_types=["sklearn.tree._tree.Tree",
                                     "xgboost.core.Booster",
                                     "xgboost.sklearn.XGBRegressor"],
            )

        mlflow.log_metrics({
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2": r2,
            "val_nasa_score": nasa,
            "training_time": elapsed,
        })

        print(f"[{model_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | "
              f"R2: {r2:.2f} | NASA Score: {nasa:.2f}")