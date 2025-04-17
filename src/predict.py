import pandas as pd
import joblib
from src.preprocessing import preprocess_data

def predict_obesity(user_input: dict) -> str:
    model = joblib.load("models/model.pkl")
    encoders = joblib.load("models/encoders.pkl")

    df = pd.DataFrame([user_input])

    for col in ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = pd.to_numeric(df[col])

    X, _, _ = preprocess_data(df, is_train=False, encoders=encoders)

    prediction = model.predict(X)
    return encoders["NObeyesdad"].inverse_transform(prediction)[0]
