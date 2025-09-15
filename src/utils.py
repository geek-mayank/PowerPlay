import os
import sys
import pickle
import numpy as np
import time

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException


def save_object(file_path, obj):
    """Save object as pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE implementation."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models (no hyperparameter tuning).
    Returns dict with R², RMSE, MAE, MAPE, and train vs test metrics.
    """
    try:
        report = {}

        for model_name, model in models.items():
            start = time.time()

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)
            mape = mean_absolute_percentage_error(y_test, y_test_pred)

            end = time.time()

            report[model_name] = {
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE": round(mape, 2),
                "time_taken_sec": round(end - start, 2),
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load pickled object."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
