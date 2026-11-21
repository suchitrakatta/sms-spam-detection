from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        message = request.form["message"]

        # Convert message into numerical data
        data = vectorizer.transform([message])

        # Predict the message
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Spam Message"
        else:
            result = "Not Spam Message"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)