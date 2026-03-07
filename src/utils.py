import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):

    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, threshold=0.6):

    try:

        report = {}

        for model_name, model in models.items():

            params = param[model_name]

        
            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            best_model.fit(X_train, y_train)

            
            if hasattr(best_model, "predict_proba"):

                y_train_prob = best_model.predict_proba(X_train)[:, 1]
                y_test_prob = best_model.predict_proba(X_test)[:, 1]

                y_train_pred = (y_train_prob >= threshold).astype(int)
                y_test_pred = (y_test_prob >= threshold).astype(int)

            else:

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                y_train_prob = y_train_pred
                y_test_prob = y_test_pred

            # Metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_auc = roc_auc_score(y_train, y_train_prob)

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_prob)

            report[model_name] = {

                "model": best_model,
                "best_params": gs.best_params_,

                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,

                "train_precision": train_precision,
                "test_precision": test_precision,

                "train_recall": train_recall,
                "test_recall": test_recall,

                "train_f1": train_f1,
                "test_f1": test_f1,

                "train_auc": train_auc,
                "test_auc": test_auc
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)