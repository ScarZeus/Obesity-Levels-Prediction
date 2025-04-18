from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os

def train_model(X, y, save_dir="models"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Accuracy graph
    plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc], color=["green", "blue"])
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.savefig("models/accuracy_graph.png")
    plt.close()

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
