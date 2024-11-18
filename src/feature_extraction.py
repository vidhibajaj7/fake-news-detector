from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(data, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data['content'])
    return X, vectorizer