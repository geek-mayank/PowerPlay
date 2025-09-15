import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple regressors (no hyperparameter tuning),
        evaluate on test_array, select best model, save it.
        Returns best_model_name and full metrics dict.
        """
        try:
            logging.info("Splitting training and test input/target")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models (default params, much faster)
            models = {
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                #"Decision Tree": DecisionTreeRegressor(random_state=42),
                #"Gradient Boosting": GradientBoostingRegressor(random_state=42),
                #"Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
                #"AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            logging.info("Starting model evaluation (no hyperparameter tuning)...")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            # Get best model by test R²
            best_model_name = max(model_report, key=lambda k: model_report[k]["test_r2"])
            best_metrics = model_report[best_model_name]

            logging.info(
                f"Best Model: {best_model_name} | "
                f"R2 = {best_metrics['test_r2']:.4f}, "
                f"RMSE = {best_metrics['RMSE']:.4f}, "
                f"MAE = {best_metrics['MAE']:.4f}, "
                f"MAPE = {best_metrics['MAPE']:.4f}"
            )

            if best_metrics["test_r2"] < 0.6:
                logging.warning("No model reached R2 > 0.6, consider tuning parameters or features")

            # Fit best model fully
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_metrics

        except Exception as e:
            raise CustomException(e, sys)
