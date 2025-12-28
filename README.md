# SMS Spam Analytics

A data analytics project demonstrating **Pandas** and **NumPy** through analysis of the UCI SMS Spam Collection dataset.

## Overview

This project analyzes the SMS Spam Collection dataset to identify patterns and characteristics that distinguish spam messages from legitimate (ham) messages. All analysis is performed using only Pandas and NumPy.

## Dataset

**SMS Spam Collection** from UCI Machine Learning Repository
- 5,574 SMS messages
- Binary labels: 'ham' (legitimate) or 'spam'
- Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

### Download Instructions

1. Download from: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
2. Extract 'SMSSpamCollection' file
3. Place it in the 'data/' folder

## Features

- **Basic Statistics** - Message counts, averages by label
- **Feature Engineering** - Length, word count, uppercase ratio, special characters
- **Length Distribution** - Analyze message lengths across spam/ham
- **Spam Indicators** - Identify keywords strongly associated with spam
- **Word Frequency** - Top words in spam vs ham messages
- **Percentile Analysis** - Statistical distribution of message lengths

## Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| Python  | 3.8+    | Core language |
| Pandas  | 2.0+    | Data manipulation & analysis |
| NumPy   | 1.24+   | Numerical operations |

## Project Structure

```
message-analytics/
├── README.md
├── requirements.txt
├── data/
│   └── SMSSpamCollection
└── src/
    ├── data_loader.py
    ├── analytics.py
    └── main.py
```

## Getting Started

### Prerequisites

```bash
pip install pandas numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
cd src
python main.py
```

## Sample Output

```
============================================================
  DATASET SUMMARY
============================================================
Total Messages:     5,574
Ham (legitimate):   4,827 (86.6%)
Spam:               747 (13.4%)
Avg Spam Length:    138.67 characters
Avg Ham Length:     71.48 characters
```

## Pandas/NumPy Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Data loading | 'pd.read_csv()' with custom separators |
| String operations | '.str.len()', '.str.contains()', '.str.count()' |
| GroupBy aggregations | '.groupby().agg()' |
| Binning | 'pd.cut()' |
| Pivot tables | '.pivot()', '.unstack()' |
| Percentiles | 'np.percentile()' |
| Boolean indexing | Filtering by condition |
| Statistical functions | '.mean()', '.std()', '.value_counts()' |

## Citation

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## Author

**Din**

## License

This project is open source and available under the [MIT License](LICENSE).
