import pandas as pd
from data_loader import load_sms_data
from analytics import (
    get_basic_stats,
    get_length_distribution,
    get_feature_comparison,
    get_word_frequency,
    get_percentile_analysis,
    generate_summary
)

def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_section("LOADING SMS SPAM COLLECTION DATASET")
    df = load_sms_data()
    print(f"Loaded {len(df):,} messages")
    print("\nSample messages:")
    print(df[['label', 'message']].head(5).to_string(index=False))
    
    print_section("DATASET SUMMARY")
    summary = generate_summary(df)
    print(f"Total Messages:     {summary['total_messages']:,}")
    print(f"Ham (legitimate):   {summary['ham_count']:,} ({100 - summary['spam_percentage']}%)")
    print(f"Spam:               {summary['spam_count']:,} ({summary['spam_percentage']}%)")
    print(f"Avg Spam Length:    {summary['avg_spam_length']} characters")
    print(f"Avg Ham Length:     {summary['avg_ham_length']} characters")
    print(f"Message Range:      {summary['min_message_length']} - {summary['max_message_length']} characters")
    
    print_section("BASIC STATISTICS BY LABEL")
    basic_stats = get_basic_stats(df)
    print(basic_stats.to_string(index=False))
    
    print_section("FEATURE COMPARISON: HAM vs SPAM")
    comparison = get_feature_comparison(df)
    print(comparison.to_string(index=False))
    
    print_section("MESSAGE LENGTH DISTRIBUTION")
    length_dist = get_length_distribution(df)
    print(length_dist.to_string(index=False))
    
    print_section("PERCENTILE ANALYSIS (Message Length)")
    percentiles = get_percentile_analysis(df)
    print(percentiles.to_string(index=False))
    
    print_section("TOP 15 WORDS IN SPAM MESSAGES")
    spam_words = get_word_frequency(df, label='spam', top_n=15)
    print(spam_words.to_string(index=False))
    
    print_section("TOP 15 WORDS IN HAM MESSAGES")
    ham_words = get_word_frequency(df, label='ham', top_n=15)
    print(ham_words.to_string(index=False))
    
    print_section("KEY INSIGHTS")
    
    print(f"1. Dataset is imbalanced: {summary['spam_percentage']}% spam vs {100-summary['spam_percentage']}% ham")
    
    comp = get_feature_comparison(df)
    length_diff = comp[comp['feature'] == 'message_length']['difference'].values[0]
    print(f"2. Spam messages are {'longer' if length_diff > 0 else 'shorter'} on average ({abs(length_diff):.1f} chars difference)")
    
    uppercase_diff = comp[comp['feature'] == 'uppercase_ratio']['difference'].values[0]
    print(f"3. Spam uses {'more' if uppercase_diff > 0 else 'less'} uppercase ({uppercase_diff:.3f} ratio difference)")
    
    exclaim_diff = comp[comp['feature'] == 'exclamation_count']['difference'].values[0]
    print(f"4. Spam has {exclaim_diff:.1f} more exclamation marks on average")
    
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()