import pandas as pd
from config.config import config
from pathlib import Path
import json


if __name__ == "__main__":
    df_risk = pd.read_parquet(config["pp"]["rbd_pred_diag"])

    thresholds = {}
    for type, path in  config["pp"]["thresholds"].items():
        dir = config["pp"]["thresholds"]
        with open(path, 'r') as file:
            thresholds[type] = json.load(file)
