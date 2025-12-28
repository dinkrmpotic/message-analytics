import numpy as np
import pandas as pd

def load_sms_data(filepath: str = "../data/SMSSpamCollection") -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['label', 'message'],
        encoding='utf-8'
    )
    
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    df['has_numbers'] = df['message'].str.contains(r'\d').astype(int)
    df['has_currency'] = df['message'].str.contains(r'[$£€]').astype(int)
    df['uppercase_ratio'] = df['message'].apply(_calculate_uppercase_ratio)
    df['exclamation_count'] = df['message'].str.count('!')
    df['question_count'] = df['message'].str.count(r'\?')
    
    return df

def _calculate_uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return round(uppercase / len(letters), 4)

if __name__ == "__main__":
    df = load_sms_data()
    print("Data loaded:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
