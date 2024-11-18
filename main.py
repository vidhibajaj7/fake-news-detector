from src.data_preprocessing import load_data, preprocess_data
from src.feature_extraction import vectorize_text
from src.model_training import train_model
from src.prediction import predict_fake_news

# Load and preprocess data
data = load_data('data/fake_news.csv')
data = preprocess_data(data)
X, vectorizer = vectorize_text(data)
y = data['label']

# Train model
model, accuracy, report = train_model(X, y)
print(f"Model Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Make a sample prediction
sample_text = "This is a sample news article to test."
print(f"Prediction for sample text: {predict_fake_news(model, vectorizer, sample_text)}")