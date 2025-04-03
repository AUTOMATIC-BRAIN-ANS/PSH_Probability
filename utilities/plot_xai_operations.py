import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def save_and_get_shap_importance(explainer, X_train, save_path, iteration):
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"{save_path}shap_summary_{iteration}.pdf")
    plt.close()

    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(importance, index=X_train.columns).sort_values(
        ascending=False
    )
    return feature_importance


def save_and_get_feature_importance_for_xgboost(model, save_path, iteration):
    importance = model.get_booster().get_score(importance_type="weight")
    feature_importance = pd.Series(importance).sort_values(ascending=False).head(10)

    xgb.plot_importance(
        model, importance_type="weight", max_num_features=10, show_values=False
    )
    plt.savefig(
        f"{save_path}feature_importance_{iteration}.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{save_path}feature_importance_{iteration}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    return feature_importance


def save_and_get_feature_importance_for_logistic_regression(
    model, X_train, save_path, iteration
):
    importance = np.abs(model.coef_[0])
    feature_importance = (
        pd.Series(importance, index=X_train.columns)
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 8))
    feature_importance.plot(kind="bar")
    plt.ylabel("Absolute Coefficient Value")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        f"{save_path}feature_importance_{iteration}.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{save_path}feature_importance_{iteration}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    return feature_importance
