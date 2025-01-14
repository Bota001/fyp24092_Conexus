import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Map labels: ham -> 0, spam -> 1
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    texts = list(data['text'])
    labels = list(data['label'])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    class SpamDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are integers
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SpamDataset(train_encodings, train_labels)
    test_dataset = SpamDataset(test_encodings, test_labels)

    return train_dataset, test_dataset
