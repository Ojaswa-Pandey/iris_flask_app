from flask import Flask, render_template, request
import joblib  # or use pickle
import numpy as np

# Load saved model and encoder
model = joblib.load("iris_model.joblib")
encoder = joblib.load("label_encoder.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get and convert input
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])
            sample = np.array([[sl, sw, pl, pw]])

            # Predict
            pred_class = model.predict(sample)[0]
            prediction = encoder.classes_[pred_class]

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
