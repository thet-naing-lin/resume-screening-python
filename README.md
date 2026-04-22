# Resume Screening Tool — Python NLP Service

Flask-based NLP microservice for scoring resumes against job descriptions. Uses TF-IDF keyword matching and Sentence-BERT semantic similarity to calculate a weighted final score.

## Tech Stack

- **Python 3.9+**
- **Flask** — lightweight REST API
- **scikit-learn** — TF-IDF vectorisation and cosine similarity
- **sentence-transformers** — Semantic similarity via `all-MiniLM-L6-v2`
- **NLTK** — Text preprocessing (stopwords, stemming)

## Scoring Algorithm

The final score is a weighted combination of two methods:

| Method   | Weight | Description                                                   |
| -------- | ------ | ------------------------------------------------------------- |
| TF-IDF   | 40%    | Keyword frequency matching between resume and job description |
| Semantic | 60%    | Sentence-BERT embedding similarity (context-aware)            |

- final_score = (tfidf_score × 0.4) + (semantic_score × 0.6)

- Scores are normalised to a 0–100 range.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone or navigate to the python service folder
cd resume-screening-python

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Running the Service

```bash
# Development
python app.py

# Production (using gunicorn)
gunicorn -w 2 -b 0.0.0.0:5001 app:app
```

Service runs at `http://127.0.0.1:5001`.

## API Endpoint

### Score a Resume (compute TF-IDF)

POST /score/tfidf

**Request Body:**

```json
{
  "resume_text": "Experienced full stack developer with 3 years in Laravel, React...",
  "job_text": "We are looking for a PHP developer with Laravel experience..."
}
```

**Response:**

```json
{
  "tfidf_score": 33.52
}
```

**Error Response:**

```json
{
  "error": "job_text is required and cannot be empty."
}
```

### Score a Resume (compute semantic)

POST /score/semantic

**Request Body:**

```json
{
  "resume_text": "Experienced full stack developer with 3 years in Laravel, React...",
  "job_text": "We are looking for a PHP developer with Laravel experience..."
}
```

**Response:**

```json
{
  "semantic_score": 59.73
}
```

**Error Response:**

```json
{
  "error": "job_text is required and cannot be empty."
}
```

## Requirements

```txt
flask
scikit-learn
sentence-transformers
nltk
numpy
torch
```

Install all at once:

```bash
pip install -r requirements.txt
```

> ⚠️ First run will download the `all-MiniLM-L6-v2` model (~90MB). Subsequent runs use the cached version.

## Environment Variables

```env
FLASK_ENV=development
FLASK_PORT=5001
MODEL_NAME=all-MiniLM-L6-v2
```

## Notes

- The Sentence-BERT model is downloaded automatically on first use via HuggingFace
- Keep the service running alongside Laravel — the backend calls it synchronously inside the queue job
- For production, run behind a process manager like `supervisor` or `systemd`
