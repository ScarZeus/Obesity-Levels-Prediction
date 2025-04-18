from sklearn.ensemble import RandomForestClassifier

def get_model():
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return clf