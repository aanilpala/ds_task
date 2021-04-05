import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from pathlib import Path


def save_actual_predicted_plot(
    predicted: np.array, actual: np.array, report_dir: str, segment: str = "inflow_new"
):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Predicted vs Actual")
    ax.set_ylabel("Predicted Inflow")
    ax.set_xlabel("Actual Inflow")
    sns.scatterplot(x=actual, y=predicted, ax=ax)
    sns.lineplot(x=actual, y=actual, ax=ax, linestyle="--", color="r")
    fig.savefig(str(Path(report_dir) / f"predicted_vs_actual-{segment}.png"))


def save_residual_plot(
    predicted: np.array, actual: np.array, report_dir: str, segment: str = "inflow_new"
):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Residual Plot")
    ax.set_ylabel("Residual")
    ax.set_xlabel("Actual Inflow")
    sns.scatterplot(x=actual, y=predicted - actual, ax=ax)
    sns.lineplot(x=actual, y=np.zeros(actual.shape), ax=ax, linestyle="--", color="r")
    fig.savefig(str(Path(report_dir) / f"residual-{segment}.png"))


def save_prediction_bound_size_boxplot(
    main_segment: str, segments_to_pred: Dict[str, Dict[str, np.array]], report_dir: str
):

    df_bound_size = pd.DataFrame(columns=["segment", "bound_size"])

    for segment, preds_dict in segments_to_pred.items():
        df = pd.DataFrame({"bound_size": preds_dict["pred_hi"] - preds_dict["pred_lo"]})
        df["segment"] = segment
        df_bound_size = df_bound_size.append(df)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_title(f"Prediction Bound Size Box-Whisker Plot - {main_segment}")
    sns.boxplot(x="segment", y="bound_size", data=df_bound_size, ax=ax)
    fig.savefig(
        str(Path(report_dir) / f"prediction_bound_size_boxplot_{main_segment}.png")
    )


def save_eval_metrics_df(
    main_segment: str, segments_to_pred: Dict[str, Dict[str, np.array]], report_dir: str
):

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    eval_metrics_df = pd.DataFrame(columns=["segment", "rmse", "r2", "mae", "bound_coverage"])

    for segment, preds_dict in segments_to_pred.items():

        eval_metrics_dict = {
            "segment": segment,
            "rmse": mean_squared_error(
                preds_dict["target"], preds_dict["pred"], squared=False
            ),
            "r2": r2_score(preds_dict["target"], preds_dict["pred"]),
            "mae": mean_absolute_error(preds_dict["target"], preds_dict["pred"]),
            "bound_coverage": (
                (preds_dict["target"] >= preds_dict["pred_lo"])
                & (preds_dict["target"] <= preds_dict["pred_hi"])
            ).mean(),
        }

        eval_metrics_df = eval_metrics_df.append(eval_metrics_dict, ignore_index=True)

    eval_metrics_df.set_index("segment", inplace=True)
    eval_metrics_df.to_csv(str(Path(report_dir) / f"{main_segment}_eval_metrics.csv"))
