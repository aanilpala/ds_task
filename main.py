import click
import pandas as pd
from pathlib import Path
import joblib
from lib.utils import extract_features, extract_targets, predict, evaluate


@click.command()
@click.option(
    "-i",
    "in_file",
    required=True,
    help="Path to csv file to be processed.",
)
@click.option(
    "-o",
    "out_folder",
    default="./data/output/",
    help="Path to the folder under which the predictions and plots are stored",
)
@click.option(
    "-m",
    "model_folder",
    default="./data/model/",
    help="Path to the folder under which the model assets are stored",
)
def batch_predict(in_file: str, model_folder: str, out_folder: str):
    """Processes the input file using the model assets stores under
    model_path and stores the result under output folder.
    """

    assert Path(model_folder).exists()

    df_transactions = pd.read_csv(in_file)
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    df_features = extract_features(df_transactions, model_folder=model_folder)
    df_preds = predict(df_features, model_folder=model_folder)
    df_targets = extract_targets(df_transactions, model_folder=model_folder)

    df_preds_w_targets = df_preds.join(df_targets)
    df_preds_w_targets.to_csv(str(Path(out_folder) / "predictions.csv"))

    evaluate(df_preds_w_targets, out_folder=out_folder)


if __name__ == "__main__":
    batch_predict()
