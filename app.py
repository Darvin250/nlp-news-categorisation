from flask import Flask, render_template, request
import joblib
import re
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]

        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])

        # Prediction
        result = model.predict(vector)[0]

        # SAFE confidence calculation (no predict_proba)
        score = model.decision_function(vector)[0]
        confidence = round((1 / (1 + np.exp(-abs(score)))) * 100, 2)

        if result == 1:
            prediction = "REAL NEWS "
        else:
            prediction = "FAKE NEWS "

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )

@app.route("/fake-news")
def fake_news():
    return render_template("fake-news.html")

if __name__ == "__main__":
    app.run(debug=True)
