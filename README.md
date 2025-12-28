# SMS Spam Detector

A complete SMS spam detection system with interactive analytics, machine learning classification, and a REST API for real-time predictions.

## 🎯 Overview

This project analyzes the UCI SMS Spam Collection dataset to:
1. **Explore** spam patterns with interactive Altair visualizations
2. **Train** a machine learning model using sentence embeddings
3. **Deploy** a FastAPI service for real-time spam detection via Docker

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SMS Message   │────▶│ Sentence-BERT    │────▶│ Logistic        │
│   (text input)  │     │ (384-dim vector) │     │ Regression      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ Spam Probability│
                                                 │   (0-100%)      │
                                                 └─────────────────┘
```

### How It Works

1. **Text Embedding**: Each message is converted to a 384-dimensional vector using `all-MiniLM-L6-v2` (a lightweight Sentence-BERT model). This captures semantic meaning - similar messages have similar vectors.

2. **Classification**: A Logistic Regression model trained on these embeddings predicts spam probability. It outputs calibrated probabilities (not just yes/no).

3. **API**: FastAPI serves predictions via REST endpoints, packaged in Docker for easy deployment.

## 📊 Dataset

**SMS Spam Collection** from UCI Machine Learning Repository
- **5,574** SMS messages
- **86.6%** ham (legitimate) / **13.4%** spam
- Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 97.73% |
| **Precision (spam)** | 98% |
| **Recall (spam)** | 88% |
| **F1-Score (spam)** | 93% |

## 📁 Project Structure

```
message-analytics/
├── README.md
├── Dockerfile              # Container definition
├── pyproject.toml          # Dependencies (UV/pip)
├── data/
│   └── SMSSpamCollection   # Dataset
├── models/
│   └── spam_classifier.joblib  # Trained model
└── src/
    ├── sms_spam_analytics.ipynb  # Analysis + training notebook
    └── api/
        └── main.py         # FastAPI application
```

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Build the image
docker build -t spam-detector .

# Run the container
docker run -p 8000:8000 spam-detector

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a FREE prize!"}'
```

### Option 2: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook for analysis
jupyter notebook src/sms_spam_analytics.ipynb

# Start the API
uvicorn src.api.main:app --reload
```

## 🔌 API Endpoints

### Health Check
```http
GET /health
```
Response: `{"status": "healthy"}`

### Predict Spam
```http
POST /predict
Content-Type: application/json

{"message": "Your text message here"}
```

Response:
```json
{
  "message": "Your text message here",
  "spam_probability": 92.36,
  "is_spam": true
}
```

### Interactive Docs
Visit `http://localhost:8000/docs` for Swagger UI

## 📈 Key Findings from Analysis

| Indicator | Spam vs Ham Ratio |
|-----------|-------------------|
| Message Length | 1.94x longer |
| Uppercase Ratio | 3.37x higher |
| Exclamation Marks | 3.53x more |
| Currency Symbols | 18.7% of spam vs 0.2% ham |

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| Analytics | Pandas, NumPy, Altair |
| ML Model | Sentence-Transformers, Scikit-learn |
| API | FastAPI, Uvicorn |
| Container | Docker, UV (fast Python packaging) |

## 📓 Notebook Contents

The Jupyter notebook includes:
1. **Data Loading & Feature Engineering**
2. **Distribution Visualizations** (pie charts, histograms, box plots)
3. **Feature Comparison** (grouped bar charts, scatter plots)
4. **Word Frequency Analysis**
5. **Spam Indicator Analysis** (uppercase, currency, exclamation marks)
6. **ML Model Training & Evaluation**

## 🔧 Development

```bash
# Install dev dependencies
pip install pandas numpy altair sentence-transformers scikit-learn fastapi uvicorn

# Train a new model (run notebook cells)
jupyter notebook src/sms_spam_analytics.ipynb

# Test API locally
uvicorn src.api.main:app --reload --port 8000
```

## 📚 Citation

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## 👤 Author

**Din**

## 📄 License

MIT License
