import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully with result!")


if __name__ == "__main__":
    train_model()
