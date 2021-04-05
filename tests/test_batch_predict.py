import pytest
import pandas as pd
from pathlib import Path
from lib.utils import extract_features, extract_targets, predict, evaluate


@pytest.fixture
def in_file():
    return "tests/data/input/sample_transactions_data.csv"


@pytest.fixture
def out_folder():
    return "tests/data/output/"


@pytest.fixture
def predictions_file(out_folder):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    return str(Path(out_folder) / "predictions.csv")


@pytest.fixture
def model_folder():
    return "data/model/"


@pytest.mark.run(order=1)
def test_extract_predict_runs(in_file, model_folder, predictions_file):

    df_transactions = pd.read_csv(in_file)
    df_features = extract_features(df_transactions, model_folder=model_folder)
    df_preds = predict(df_features, model_folder=model_folder)
    df_targets = extract_targets(df_transactions, model_folder=model_folder)

    df_preds_w_targets = df_preds.join(df_targets)
    df_preds_w_targets.to_csv(predictions_file)


@pytest.mark.run(order=2)
def test_evaluate_runs(predictions_file, out_folder):

    df_preds_w_targets = pd.read_csv(predictions_file)
    evaluate(df_preds_w_targets, out_folder=out_folder)
