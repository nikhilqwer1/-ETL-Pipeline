# Train model
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier().fit(X, y)
joblib.dump(model, "model.pkl")
# Flask API (app.py)
from flask import Flask, request, jsonify
import joblib

model = joblib.load("model.pkl")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json['features']
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run()
