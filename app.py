from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question1 = request.form['question1']
        question2 = request.form['question2']
        
        # Add your backend logic here to generate the answer based on question1 and question2
        
        answer = f"The answer to {question1} and {question2} is ..."

        return render_template('index.html', answer=answer)

    return render_template('index.html')


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        review = request.form['review']
        
        # Load the pre-trained ML model
        model = joblib.load('sentiment_model.pkl')
        
        # Preprocess the review text (assuming you have a similar preprocessing step)
        preprocessed_review = preprocess(review)
        
        # Vectorize the preprocessed review text
        vectorizer = CountVectorizer(vocabulary=joblib.load('vectorizer.pkl'))
        vectorized_review = vectorizer.transform([preprocessed_review])
        
        # Predict the sentiment using the ML model
        sentiment = model.predict(vectorized_review)[0]
        
        return render_template('sentiment.html', review=review, sentiment=sentiment)

    return render_template('sentiment.html')

def preprocess(review):
    # Add your preprocessing steps here
    preprocessed_review = review.lower()  # Example: convert to lowercase

    return preprocessed_review

if __name__ == '__main__':
    app.run(debug=True)
