from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle
import os

def preprocess_data(df, save_dir="models"):
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save encoders and scaler
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/encoder.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{save_dir}/target_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)

    return X_scaled, y
