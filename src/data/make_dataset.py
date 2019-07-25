import os

import pandas as pd

from config import PROJECT_DIR


def get_raw_data():
    column_names = ["Recency", "Frequency", "Amount", "Times", "Donation"]
    path = os.path.join(PROJECT_DIR, "data", "raw", "transfusion.data")
    return pd.read_csv(path, names=column_names, header=0)


def save_processed_data(data):
    path = os.path.join(PROJECT_DIR, "data", "processed", "transfusion_1.csv")
    data.to_csv(path, index=False)


def get_processed_data(filename):
    path = os.path.join(PROJECT_DIR, "data", "processed", filename)
    return pd.read_csv(path)
