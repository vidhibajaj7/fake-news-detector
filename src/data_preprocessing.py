import pandas as pd
import re

def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna(subset=['title', 'text', 'label'])
    data['content'] = data['title'] + " " + data['text']
    return data

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def preprocess_data(data):
    data['content'] = data['content'].apply(preprocess_text)
    return data