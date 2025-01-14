from transformers import BertTokenizer

def initialize_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
