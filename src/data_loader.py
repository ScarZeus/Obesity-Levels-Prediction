import pandas as pd

def load_dataset(path="dataSet\obesityLevel.csv"):
    return pd.read_csv(path)
