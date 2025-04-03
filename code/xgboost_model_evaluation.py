import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utilities.df_operations import load_dataset, setup_directories
from utilities.model_operations import calculate_metrics, save_roc_auc_figure
from utilities.plot_xai_operations import (
    save_and_get_feature_importance_for_xgboost,
    save_and_get_shap_importance,
)

ROOT_PATH = Path("")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_predict(X_train, y_train, X_test, param_comb, eval_metric="auc"):
    model = xgb.XGBClassifier(
        eval_metric=eval_metric, objective="binary:logistic", **param_comb
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, predictions, y_proba, param_comb


def evaluate_with_kfold(metadata, label, base_path, file_path, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aggregate_results = []
    iteration_counter = 0

    model_dir = setup_directories(
        base_path, f"xgboost_kfold_{file_path.name}"
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

    param_grid = {
        "max_depth": [3, 4, 5],
        "gamma": [0, 0.1, 0.5, 1, 5],
        "subsample": [0.5, 0.75, 1],
        "colsample_bytree": [0.5, 0.75, 1],
        "reg_lambda": [1, 1.5],
    }

    param_combinations = list(
        itertools.product(
            param_grid["max_depth"],
            param_grid["gamma"],
            param_grid["subsample"],
            param_grid["colsample_bytree"],
            param_grid["reg_lambda"],
        )
    )

    labels = metadata[label]

    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    all_shap_importances = {}
    all_xgb_importances = {}
    all_fold_results = {}
    all_param_comb_list = {}

    def save_progress():
        # with open(f'{model_dir}/fold_results.json', 'w') as f:
        #     json.dump(all_fold_results, f, indent=4)
        # with open(f'{model_dir}/param_comb_list.json', 'w') as f:
        #     json.dump(all_param_comb_list, f, indent=4)
        # with open(f'{model_dir}/shap_importances.json', 'w') as f:
        #     json.dump(all_shap_importances, f, indent=4)
        # with open(f'{model_dir}/xgb_importances.json', 'w') as f:
        #     json.dump(all_xgb_importances, f, indent=4)
        # with open(f'{model_dir}/aggregate_results.json', 'w') as f:
        #     json.dump(aggregate_results, f, indent=4)
        pass

    for param_comb in param_combinations:
        param_dict = {
            "max_depth": param_comb[0],
            "gamma": param_comb[1],
            "subsample": param_comb[2],
            "colsample_bytree": param_comb[3],
            "reg_lambda": param_comb[4],
        }
        print(f"Evaluating parameters: {param_dict}")

        metrics_list = []
        dict_shap_importances = {}
        dict_xgb_importances = {}
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

            explainer = shap.Explainer(model, X_train)
            dict_shap_importances[fold_counter] = save_and_get_shap_importance(
                explainer, X_train, save_path, fold_counter
            ).to_dict()  # utility for generating shap importance

            dict_xgb_importances[fold_counter] = (
                save_and_get_feature_importance_for_xgboost(
                    model, save_path, fold_counter
                ).to_dict()
            )  # utility for generating feature importance

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
        all_xgb_importances[iteration_counter] = dict_xgb_importances

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
