from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1
        else:
            score += np.exp(diff / 10.0) - 1
    return score

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)
    return rmse, mae, r2, score