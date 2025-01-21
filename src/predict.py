import pickle


def predict(sample):
    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Make prediction
    prediction = model.predict([sample])
    return prediction


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]  # Example input
    print(f"Prediction: {predict(sample)}")
