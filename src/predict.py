import pickle
import numpy as np

def load_all_models(model_dir="models"):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    return model, encoders, scaler, target_encoder

def predict_user_input(user_input_dict):
    model, encoders, scaler, target_encoder = load_all_models()

    for key in user_input_dict:
        if key in encoders:
            user_input_dict[key] = encoders[key].transform([user_input_dict[key]])[0]

    values = np.array([list(user_input_dict.values())]).astype(float)
    scaled = scaler.transform(values)
    pred = model.predict(scaled)
    return target_encoder.inverse_transform(pred)[0]
