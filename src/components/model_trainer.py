import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


class ModeltrainerConfig:
    trained_model_file_path = os.path.join ("artifacts","model.pkl")
    
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1]
                },
                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGB Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost Regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                }
            }

            best_model = None
            best_model_score = float("-inf")
            best_model_name = ""

            for model_name, model in models.items():

                logging.info(f"Tuning model: {model_name}")

                if params[model_name]:
                    gs = GridSearchCV(
                        model,
                        params[model_name],
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    gs.fit(X_train, y_train)
                    tuned_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    tuned_model = model

                y_test_pred = tuned_model.predict(X_test)
                test_score = r2_score(y_test, y_test_pred)

                if test_score > best_model_score:
                    best_model_score = test_score
                    best_model = tuned_model
                    best_model_name = model_name

            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)