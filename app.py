from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__) 

feature_names = [
    "age","sex","chest pain type","resting bp s","cholesterol",
    "fasting blood sugar","resting ecg","max heart rate",
    "exercise angina","oldpeak","ST slope"
]

lr_model = pickle.load(open("Model_LogisticRegression.pkl", "rb"))
rf_model = pickle.load(open("Model_RandomForest.pkl", "rb"))

tree_img = "Gambar Tree.png"  # ⬅️ BENAR

@app.route("/")
def index():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array([[float(request.form[f]) for f in feature_names]])
    selected_model = request.form["model"]

    if selected_model == "lr":
        pred = lr_model.predict(data)[0]
        model_name = "Logistic Regression"
        img = None
    else:
        pred = rf_model.predict(data)[0]
        model_name = "Random Forest"
        img = tree_img

    return render_template(
        "index.html",
        selected_model=model_name,
        prediction=pred,
        tree_img=img
    )

if __name__ == "__main__":
    app.run(debug=True)
