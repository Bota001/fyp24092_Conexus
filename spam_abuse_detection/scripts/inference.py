import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def load_model():
    tokenizer = BertTokenizer.from_pretrained('./models/trained_model')
    model = BertForSequenceClassification.from_pretrained('./models/trained_model')
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

if __name__ == "__main__":
    classifier = load_model()
    test_comments = [
        "You won a free ticket to Hawaii! Click here!",
        "This is a spam-free message.",
        "Your account has been compromised. Please reset your password immediately.",
    ]
    predictions = classifier(test_comments)
    for comment, prediction in zip(test_comments, predictions):
        print(f"Comment: {comment}")
        print(f"Prediction: {prediction['label']}, Score: {prediction['score']:.4f}")
