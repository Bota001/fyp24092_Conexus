import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from utils.data_loader import load_data

# Load the dataset
train_dataset, test_dataset = load_data("data/processed/preprocessed_data.csv")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./models/trained_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./models/trained_model')
tokenizer.save_pretrained('./models/trained_model')
