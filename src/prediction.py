from src.data_preprocessing import preprocess_text

def predict_fake_news(model, vectorizer, text):
    # Preprocess the input text
    text_processed = preprocess_text(text)
    
    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([text_processed])
    
    # Predict using the trained model
    prediction = model.predict(text_vectorized)
    
    # Return result based on the prediction
    return "Real" if prediction[0] == 1 else "Fake"