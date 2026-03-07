import os
import sys

import mlflow
import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='badishanarendar123', repo_name='_mlFlow', mlflow=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models


class ModelTrainer:

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

        

            
            models = {

                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            }

            params = {

                "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},

                "Random Forest": {
                    "n_estimators": [200, 300],
                    "max_depth": [10, 15],
                    "min_samples_split": [5, 10]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05],
                    "max_depth": [3, 5]
                },

                "XGBoost": {
                    "n_estimators": [300, 500],
                    "learning_rate": [0.01, 0.05],
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 3],
                    "subsample": [0.8,0.6],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [1, 5],
                    "scale_pos_weight": [ 2, 5, 10]
                }
            }


            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params,
                threshold=0.6
            )

            mlflow.set_experiment("Loan_Prediction_Experiments")

            print("\nLogging experiments to MLflow...\n")

            for model_name, metrics in model_report.items():

                with mlflow.start_run(run_name=model_name):

                    model = metrics["model"]

                
                    mlflow.log_params(metrics["best_params"])

                    # log metrics
                    mlflow.log_metric("train_accuracy", metrics["train_accuracy"])
                    mlflow.log_metric("test_accuracy", metrics["test_accuracy"])

                    mlflow.log_metric("train_precision", metrics["train_precision"])
                    mlflow.log_metric("test_precision", metrics["test_precision"])

                    mlflow.log_metric("train_recall", metrics["train_recall"])
                    mlflow.log_metric("test_recall", metrics["test_recall"])

                    mlflow.log_metric("train_f1", metrics["train_f1"])
                    mlflow.log_metric("test_f1", metrics["test_f1"])

                    mlflow.log_metric("train_auc", metrics["train_auc"])
                    mlflow.log_metric("test_auc", metrics["test_auc"])

                    mlflow.sklearn.log_model(model, "model")

                    print(f"{model_name} logged to MLflow")

            print("\nAll experiments logged to MLflow successfully.")

        except Exception as e:
            raise CustomException(e, sys)
