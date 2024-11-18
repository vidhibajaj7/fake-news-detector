import pickle
from sklearn.ensemble import RandomForestClassifier  # Example model; use any model you prefer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('fake_news.csv')  # Ensure 'fake_news.csv' is in the same directory as this script
X = data['text']  # Replace 'text' with your actual column name for news content
y = data['label']  # Replace 'label' with the actual column name for labels (e.g., 0 for real, 1 for fake)

# Initialize and fit the vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to model.pkl
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer to vectorizer.pkl
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer have been saved successfully as 'model.pkl' and 'vectorizer.pkl'.")