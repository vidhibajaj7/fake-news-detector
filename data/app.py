import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

# Add your prediction route here (if needed)

if __name__ == '__main__':
    app.run(debug=True)

import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news article text from the form
    news_article = request.form['news']

    # Vectorize the input
    input_vectorized = vectorizer.transform([news_article])

    # Make prediction
    prediction = model.predict(input_vectorized)

    # Determine result
    result = "Fake News" if prediction[0] == 1 else "Real News"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)