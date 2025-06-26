from flask import Flask, request, send_file
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]
    result = f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}"
    return f"<h2>{result}</h2><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
