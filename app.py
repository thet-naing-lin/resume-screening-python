from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model ONCE at startup ──────────────────────────────
logger.info("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Model loaded successfully.")


# ── THIS is the missing function that caused your error ─────
def preprocess(text: str) -> str:
    """
    Clean and normalize text before scoring.
    - Lowercase everything
    - Remove special characters, keep letters/numbers/spaces
    - Collapse multiple spaces/newlines into one space
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text)             # collapse whitespace
    return text.strip()


# ── Input validator ─────────────────────────────────────────
def validate_input(data):
    if not data:
        return "Request body is empty or not JSON."
    if not data.get('resume_text', '').strip():
        return "resume_text is required and cannot be empty."
    if not data.get('job_text', '').strip():
        return "job_text is required and cannot be empty."
    return None


# ── US-011: TF-IDF ──────────────────────────────────────────
@app.route('/score/tfidf', methods=['POST'])
def tfidf_score():
    data = request.get_json()
    error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    # preprocess() is now defined — no more NameError
    resume_text = preprocess(data['resume_text'])
    job_text    = preprocess(data['job_text'])

    try:
        vectorizer   = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        score        = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        result       = round(float(score) * 100, 2)

        logger.info(f"TF-IDF score computed: {result}")
        return jsonify({'tfidf_score': result})

    except Exception as e:
        logger.error(f"TF-IDF error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ── US-012: Semantic (MiniLM-L6-V2) ─────────────────────────
@app.route('/score/semantic', methods=['POST'])
def semantic_score():
    data = request.get_json()
    error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    # preprocess() is now defined — no more NameError
    resume_text = preprocess(data['resume_text'])
    job_text    = preprocess(data['job_text'])

    try:
        embeddings = model.encode([resume_text, job_text], convert_to_numpy=True)
        score      = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        result     = round(float(score) * 100, 2)

        logger.info(f"Semantic score computed: {result}")
        return jsonify({'semantic_score': result})

    except Exception as e:
        logger.error(f"Semantic error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ── Health check ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'all-MiniLM-L6-v2'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)