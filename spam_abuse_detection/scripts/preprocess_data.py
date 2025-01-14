import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    data['text'] = data['text'].apply(preprocess_text)
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data("data/raw/spam.csv", "data/processed/preprocessed_data.csv")
