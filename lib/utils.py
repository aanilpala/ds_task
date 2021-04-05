import pandas as pd
import joblib
from pathlib import Path


def extract_features(
    df_transactions: pd.DataFrame, model_folder: str, progress_bar: bool = False
) -> pd.DataFrame:

    from .feng_utils import calc_features
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=progress_bar)

    ttype_direction_mapping, mcc_group_mapping = joblib.load(
        str(Path(model_folder) / "ttype_mcc_group_mappings.pkl")
    )
    ttypes = [*ttype_direction_mapping]
    mcc_groups = [*mcc_group_mapping]

    df_features = (
        df_transactions.groupby(["user_id"])
        .parallel_apply(lambda group: calc_features(group, ttypes, mcc_groups))
        .apply(pd.Series)
    )

    returning_user_lookup_table_df = pd.read_csv(
        str(Path(model_folder) / "returning_user_lookup_table_df.csv")
    )
    returning_user_lookup_table_df.set_index("user_id", inplace=True)

    df_features = df_features.join(returning_user_lookup_table_df)
    df_features["is_new_customer"] = df_features["prev_monthly_in_flow_avg"].isna()

    inflow_model_features = joblib.load(
        str(Path(model_folder) / "inflow_model_features.pkl")
    )
    outflow_model_features = joblib.load(
        str(Path(model_folder) / "outflow_model_features.pkl")
    )
    extracted_features = df_features.columns
    assert set(extracted_features) >= set(inflow_model_features) and set(
        extracted_features
    ) >= set(outflow_model_features)

    return df_features


def extract_targets(df_transactions: pd.DataFrame, model_folder: str) -> pd.DataFrame:

    ttype_direction_mapping, _ = joblib.load(
        str(Path(model_folder) / "ttype_mcc_group_mappings.pkl")
    )

    df_transactions["direction"] = df_transactions["transaction_type"].apply(
        ttype_direction_mapping.get
    )

    df_transactions["in_flow"] = df_transactions.apply(
        lambda tx: tx.amount_internal_currency if tx.direction == "In" else 0, axis=1
    )
    df_transactions["out_flow"] = df_transactions.apply(
        lambda tx: tx.amount_internal_currency if tx.direction == "Out" else 0, axis=1
    )

    df_targets = df_transactions.groupby("user_id").agg(
        {"in_flow": ["sum"], "out_flow": ["sum"]}
    )
    df_targets.columns = df_targets.columns.map("_".join)

    return df_targets


def predict(df: pd.DataFrame, model_folder: str) -> pd.DataFrame:

    inflow_model_pipe, inflow_model_lo_pipe, inflow_model_hi_pipe = joblib.load(
        str(Path(model_folder) / "inflow_model_pipe.pkl")
    )
    outflow_model_pipe, outflow_model_lo_pipe, outflow_model_hi_pipe = joblib.load(
        str(Path(model_folder) / "outflow_model_pipe.pkl")
    )

    inflow_model_features = joblib.load(
        str(Path(model_folder) / "inflow_model_features.pkl")
    )
    outflow_model_features = joblib.load(
        str(Path(model_folder) / "outflow_model_features.pkl")
    )

    df_preds = pd.DataFrame(
        columns=[
            "is_new_customer",
            "in_flow_sum_pred",
            "in_flow_sum_pred_lo",
            "in_flow_sum_pred_hi",
            "out_flow_sum_pred",
            "out_flow_sum_pred_lo",
            "out_flow_sum_pred_hi",
        ]
    )

    df_preds["is_new_customer"] = df["is_new_customer"]

    df_preds["in_flow_sum_pred"] = inflow_model_pipe.predict(
        df[inflow_model_features].values
    )
    df_preds["in_flow_sum_pred_lo"] = inflow_model_lo_pipe.predict(
        df[inflow_model_features].values
    )
    df_preds["in_flow_sum_pred_hi"] = inflow_model_hi_pipe.predict(
        df[inflow_model_features].values
    )

    df_preds["out_flow_sum_pred"] = outflow_model_pipe.predict(
        df[outflow_model_features].values
    )
    df_preds["out_flow_sum_pred_lo"] = outflow_model_lo_pipe.predict(
        df[outflow_model_features].values
    )
    df_preds["out_flow_sum_pred_hi"] = outflow_model_hi_pipe.predict(
        df[outflow_model_features].values
    )

    return df_preds


def evaluate(df: pd.DataFrame, out_folder: str):

    from .report_utils import (
        save_eval_metrics_df,
        save_prediction_bound_size_boxplot,
        save_actual_predicted_plot,
        save_residual_plot,
    )

    report_dir = Path(out_folder) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    df_preds_returning = df[~df["is_new_customer"]]
    df_preds_new = df[df["is_new_customer"]]
    inflow_segments_to_pred = {}
    outflow_segments_to_pred = {}

    if not df_preds_new.empty:
        inflow_segments_to_pred["new"] = {
            "target": df_preds_new["in_flow_sum"].values,
            "pred": df_preds_new["in_flow_sum_pred"].values,
            "pred_lo": df_preds_new["in_flow_sum_pred_lo"].values,
            "pred_hi": df_preds_new["in_flow_sum_pred_hi"].values,
        }

        save_actual_predicted_plot(
            predicted=df_preds_new["in_flow_sum_pred"].values,
            actual=df_preds_new["in_flow_sum"].values,
            report_dir=str(report_dir),
            segment="inflow_new",
        )

        save_residual_plot(
            predicted=df_preds_new["in_flow_sum_pred"].values,
            actual=df_preds_new["in_flow_sum"].values,
            report_dir=str(report_dir),
            segment="inflow_new",
        )

    if not df_preds_returning.empty:
        inflow_segments_to_pred["returning"] = {
            "target": df_preds_returning["in_flow_sum"].values,
            "pred": df_preds_returning["in_flow_sum_pred"].values,
            "pred_lo": df_preds_returning["in_flow_sum_pred_lo"].values,
            "pred_hi": df_preds_returning["in_flow_sum_pred_hi"].values,
        }

        save_actual_predicted_plot(
            predicted=df_preds_returning["in_flow_sum_pred"].values,
            actual=df_preds_returning["in_flow_sum"].values,
            report_dir=str(report_dir),
            segment="inflow_returning",
        )

        save_residual_plot(
            predicted=df_preds_returning["in_flow_sum_pred"].values,
            actual=df_preds_returning["in_flow_sum"].values,
            report_dir=str(report_dir),
            segment="inflow_returning",
        )

    if not df_preds_new.empty:
        outflow_segments_to_pred["new"] = {
            "target": df_preds_new["out_flow_sum"].values,
            "pred": df_preds_new["out_flow_sum_pred"].values,
            "pred_lo": df_preds_new["out_flow_sum_pred_lo"].values,
            "pred_hi": df_preds_new["out_flow_sum_pred_hi"].values,
        }

        save_actual_predicted_plot(
            predicted=df_preds_new["out_flow_sum_pred"].values,
            actual=df_preds_new["out_flow_sum"].values,
            report_dir=str(report_dir),
            segment="outflow_new",
        )

        save_residual_plot(
            predicted=df_preds_new["out_flow_sum_pred"].values,
            actual=df_preds_new["out_flow_sum"].values,
            report_dir=str(report_dir),
            segment="outflow_new",
        )

    if not df_preds_returning.empty:
        outflow_segments_to_pred["returning"] = {
            "target": df_preds_returning["out_flow_sum"].values,
            "pred": df_preds_returning["out_flow_sum_pred"].values,
            "pred_lo": df_preds_returning["out_flow_sum_pred_lo"].values,
            "pred_hi": df_preds_returning["out_flow_sum_pred_hi"].values,
        }

        save_actual_predicted_plot(
            predicted=df_preds_returning["out_flow_sum_pred"].values,
            actual=df_preds_returning["out_flow_sum"].values,
            report_dir=str(report_dir),
            segment="outflow_returning",
        )

        save_residual_plot(
            predicted=df_preds_returning["out_flow_sum_pred"].values,
            actual=df_preds_returning["out_flow_sum"].values,
            report_dir=str(report_dir),
            segment="outflow_returning",
        )

    save_eval_metrics_df(
        main_segment="inflow",
        segments_to_pred=inflow_segments_to_pred,
        report_dir=str(report_dir),
    )
    save_prediction_bound_size_boxplot(
        main_segment="inflow",
        segments_to_pred=inflow_segments_to_pred,
        report_dir=str(report_dir),
    )

    save_eval_metrics_df(
        main_segment="outflow",
        segments_to_pred=outflow_segments_to_pred,
        report_dir=str(report_dir),
    )
    save_prediction_bound_size_boxplot(
        main_segment="outflow",
        segments_to_pred=outflow_segments_to_pred,
        report_dir=str(report_dir),
    )
