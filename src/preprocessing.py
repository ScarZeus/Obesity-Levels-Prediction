from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df, is_train=True, encoders=None):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = encoders or {}

    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"] if "NObeyesdad" in df.columns else None

    return X, y, encoders
