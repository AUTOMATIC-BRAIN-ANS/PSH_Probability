import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utilities.df_operations import load_dataset, setup_directories
from utilities.model_operations import calculate_metrics, save_roc_auc_figure
from utilities.plot_xai_operations import (
    save_and_get_feature_importance_for_logistic_regression,
    save_and_get_shap_importance,
)

ROOT_PATH = Path("")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_predict(X_train, y_train, X_test, param_comb):
    model = LogisticRegression(**param_comb)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, predictions, y_proba, param_comb


def evaluate_with_kfold(metadata, label, base_path, file_path, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aggregate_results = []
    iteration_counter = 0

    model_dir = setup_directories(
        base_path, f"logistic_regression_kfold_{file_path.name}"
    )  # utility for setting up directories

    metadata.columns = metadata.columns.astype(str)
    metadata.columns = (
        metadata.columns.str.replace("[", "")
        .str.replace("]", "")
        .str.replace("<", "")
        .str.replace(">", "")
    )

    metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
    metadata.dropna(axis=1, inplace=True)
    metadata.dropna(axis=0, inplace=True)

    features = metadata.drop([label, "ID"], axis=1)
    N = len(features)
    print(features.shape, file_path.name)
    param_grid = {
        "C": [2 * N, N, 1, 1 / N, 1 / 2 * N],
        "class_weight": [None, "balanced"],
    }

    param_combinations = list(
        itertools.product(param_grid["C"], param_grid["class_weight"])
    )

    labels = metadata[label]

    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    all_shap_importances = {}
    all_log_reg_importances = {}
    all_fold_results = {}
    all_param_comb_list = {}

    def save_progress():
        # with open(f"{model_dir}/fold_results.json", "w") as f:
        #     json.dump(all_fold_results, f, indent=4)
        # with open(f"{model_dir}/param_comb_list.json", "w") as f:
        #     json.dump(all_param_comb_list, f, indent=4)
        # with open(f"{model_dir}/shap_importances.json", "w") as f:
        #     json.dump(all_shap_importances, f, indent=4)
        # with open(f"{model_dir}/log_reg_importances.json", "w") as f:
        #     json.dump(all_log_reg_importances, f, indent=4)
        # with open(f"{model_dir}/perm_importances.json", "w") as f:
        #     json.dump(all_perm_importances, f, indent=4)
        # with open(f"{model_dir}/aggregate_results.json", "w") as f:
        #     json.dump(aggregate_results, f, indent=4)
        pass

    for param_comb in param_combinations:
        param_dict = {"C": param_comb[0], "class_weight": param_comb[1]}
        print(f"Evaluating parameters: {param_dict}")

        metrics_list = []
        dict_shap_importances = {}
        dict_log_reg_importances = {}
        fold_results = {}

        fold_counter = 0
        for train_index, test_index in skf.split(features, labels):
            save_path = os.path.join(
                model_dir, f"iteration_{iteration_counter}_fold_{fold_counter}/"
            )
            os.makedirs(save_path, exist_ok=True)

            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            model, predictions, y_proba, best_params = train_and_predict(
                X_train, y_train, X_test, param_dict
            )
            metrics = calculate_metrics(y_test, predictions, y_proba)
            metrics_list.append(metrics)

            fold_results[fold_counter] = {
                "metrics": metrics,
                "params": param_dict,
                "best_params": best_params,
            }

            explainer = shap.LinearExplainer(
                model, X_train, feature_perturbation="interventional"
            )
            dict_shap_importances[fold_counter] = save_and_get_shap_importance(
                explainer, X_train, save_path, fold_counter
            ).to_dict()  # utility for generating shap importance

            dict_log_reg_importances[fold_counter] = (
                save_and_get_feature_importance_for_logistic_regression(
                    model, X_train, save_path, fold_counter
                ).to_dict()  # utility for generating feature importance
            )

            save_roc_auc_figure(
                y_test, y_proba, save_path, fold_counter
            )  # utility for generating roc auc curve

            fold_counter += 1

        aggregate_metrics = {
            "mean": pd.DataFrame(metrics_list).mean().to_dict(),
            "std": pd.DataFrame(metrics_list).std().to_dict(),
        }
        aggregate_results.append(
            {
                "iteration": iteration_counter,
                "metrics": aggregate_metrics,
                "params": param_dict,
            }
        )

        all_fold_results[iteration_counter] = fold_results
        all_param_comb_list[iteration_counter] = param_dict
        all_shap_importances[iteration_counter] = dict_shap_importances
        all_log_reg_importances[iteration_counter] = dict_log_reg_importances

        iteration_counter += 1
        save_progress()

    return aggregate_results


if __name__ == "__main__":
    metadata_signals_path = Path("")
    base_path = Path("")

    all_files = metadata_signals_path.glob("*.csv")

    for file_path in all_files:
        metadata = load_dataset(file_path)
        aggregate_results = evaluate_with_kfold(metadata, "PSH", base_path, file_path)
