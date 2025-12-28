import numpy as np
import pandas as pd

def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby('label').agg(
        count=('message', 'count'),
        avg_length=('message_length', 'mean'),
        avg_words=('word_count', 'mean'),
        avg_uppercase_ratio=('uppercase_ratio', 'mean'),
        avg_exclamations=('exclamation_count', 'mean')
    ).round(2).reset_index()
    
    stats['percentage'] = (stats['count'] / stats['count'].sum() * 100).round(2)
    
    return stats

def get_length_distribution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['length_bucket'] = pd.cut(
        df['message_length'],
        bins=[0, 50, 100, 150, 200, np.inf],
        labels=['0-50', '51-100', '101-150', '151-200', '200+']
    )
    
    distribution = df.groupby(['length_bucket', 'label'], observed=True).size().unstack(fill_value=0)
    distribution['total'] = distribution.sum(axis=1)
    distribution['spam_ratio'] = (distribution['spam'] / distribution['total'] * 100).round(2)
    
    return distribution.reset_index()

def get_feature_comparison(df: pd.DataFrame) -> pd.DataFrame:
    features = ['message_length', 'word_count', 'uppercase_ratio', 
                'exclamation_count', 'question_count', 'has_numbers', 'has_currency']
    
    comparison = []
    
    for feature in features:
        ham_mean = df[df['label'] == 'ham'][feature].mean()
        spam_mean = df[df['label'] == 'spam'][feature].mean()
        ham_std = df[df['label'] == 'ham'][feature].std()
        spam_std = df[df['label'] == 'spam'][feature].std()
        
        comparison.append({
            'feature': feature,
            'ham_mean': round(ham_mean, 3),
            'ham_std': round(ham_std, 3),
            'spam_mean': round(spam_mean, 3),
            'spam_std': round(spam_std, 3),
            'difference': round(spam_mean - ham_mean, 3)
        })
    
    return pd.DataFrame(comparison)

def get_word_frequency(df: pd.DataFrame, label: str = None, top_n: int = 20) -> pd.DataFrame:
    if label:
        texts = df[df['label'] == label]['message']
    else:
        texts = df['message']
    
    all_words = ' '.join(texts).lower().split()
    
    stop_words = {'i', 'me', 'my', 'you', 'your', 'we', 'the', 'a', 'an', 'is', 'are', 
                  'was', 'to', 'of', 'and', 'in', 'it', 'for', 'on', 'with', 'at', 'be',
                  'this', 'that', 'have', 'do', 'will', 'can', 'but', 'or', 'so', 'if',
                  'just', 'not', 'u', 'ur', 'im', 'dont', 'its', 'got', 'get', 'been'}
    
    filtered_words = [w for w in all_words if w.isalpha() and len(w) > 2 and w not in stop_words]
    
    word_counts = pd.Series(filtered_words).value_counts().head(top_n)
    
    return pd.DataFrame({'word': word_counts.index, 'count': word_counts.values})

def get_percentile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    results = []
    for label in ['ham', 'spam']:
        subset = df[df['label'] == label]['message_length']
        for p in percentiles:
            results.append({
                'label': label,
                'percentile': f'P{p}',
                'message_length': int(np.percentile(subset, p))
            })
    
    result_df = pd.DataFrame(results)
    return result_df.pivot(index='percentile', columns='label', values='message_length').reset_index()

def generate_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    spam_count = (df['label'] == 'spam').sum()
    ham_count = (df['label'] == 'ham').sum()
    
    spam_df = df[df['label'] == 'spam']
    ham_df = df[df['label'] == 'ham']
    
    return {
        'total_messages': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_percentage': round(spam_count / total * 100, 2),
        'avg_spam_length': round(spam_df['message_length'].mean(), 2),
        'avg_ham_length': round(ham_df['message_length'].mean(), 2),
        'max_message_length': df['message_length'].max(),
        'min_message_length': df['message_length'].min()
    }