import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_signals(patient_id, base_path):
    file_path = base_path / f"{patient_id}.csv"
    if file_path.exists():
        return pd.read_csv(file_path, delimiter=";", decimal=",")
    else:
        return pd.DataFrame()


def load_dataset(metadata_path):
    metadata = pd.read_csv(metadata_path, index_col=0)
    return metadata


def setup_directories(base_path, model_name):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = Path(base_path) / f"{model_name}_{date_str}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir
