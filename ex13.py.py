from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML Model API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = data["features"]

    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)

    predicted_class = int(prediction[0])

    class_names = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    return jsonify({
        "prediction": predicted_class,
        "class_name": class_names[predicted_class]
    })


if __name__ == "__main__":
    app.run(debug=True)
