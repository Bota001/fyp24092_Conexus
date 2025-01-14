from transformers import Trainer, TrainingArguments
from utils.data_loader import load_data

# Load the dataset
_, test_dataset = load_data("data/processed/preprocessed_data.csv")

# Define evaluation arguments
evaluation_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
)

# Evaluate the model
trainer = Trainer(model="./models/trained_model")
results = trainer.evaluate(eval_dataset=test_dataset)
print("Evaluation Results:", results)
