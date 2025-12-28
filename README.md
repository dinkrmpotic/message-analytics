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

1. **Text Embedding**: Each message is converted to a 384-dimensional vector using `all-MiniLM-L6-v2` (a lightweight Sentence-BERT model). This captures semantic meaning.

2. **Classification**: A Logistic Regression model trained on these embeddings predicts spam probability with calibrated outputs.

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
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── data/
│   └── SMSSpamCollection
├── models/
│   └── spam_classifier.joblib
└── src/
    ├── sms_spam_analytics.ipynb
    └── api/
        └── main.py
```

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
docker build -t spam-detector .
docker run -p 8000:8000 spam-detector
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a FREE prize!"}'
```

### Option 2: Run Locally

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```

### Predict Spam
```
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
Visit http://localhost:8000/docs for Swagger UI

## 📈 Key Findings

| Indicator | Spam vs Ham |
|-----------|-------------|
| Message Length | 1.94x longer |
| Uppercase Ratio | 3.37x higher |
| Exclamation Marks | 3.53x more |

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| Analytics | Pandas, NumPy, Altair |
| ML Model | Sentence-Transformers, Scikit-learn |
| API | FastAPI, Uvicorn |
| Container | Docker, UV |

## 📚 Citation

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## Authors

**Din Krmpotić**

## License

This project is open source and available under the [MIT License](LICENSE).
