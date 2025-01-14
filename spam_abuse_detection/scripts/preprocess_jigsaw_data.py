import pandas as pd

def preprocess_jigsaw(input_file, output_file):
    # Load the raw dataset
    data = pd.read_csv(input_file)
    
    # Define toxicity columns
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Label as spam (1) if any toxicity column has a value of 1, else ham (0)
    data['label'] = data[toxicity_columns].max(axis=1)
    data['label'] = data['label'].apply(lambda x: 1 if x == 1 else 0)
    
    # Keep only relevant columns
    data = data[['label', 'comment_text']]
    data.rename(columns={'comment_text': 'text'}, inplace=True)
    
    # Save the processed dataset
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_jigsaw("data/raw/train.csv", "data/processed/preprocessed_data.csv")
