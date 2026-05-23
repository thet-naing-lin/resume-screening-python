from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import re
import logging
import threading

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MAX_SEMANTIC_TEXT_CHARS = int(os.getenv("MAX_SEMANTIC_TEXT_CHARS", "8000"))

# Avoid extra tokenizer threads on small Render instances.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch is not None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception as e:
        logger.warning("Torch thread tuning skipped: %s", e)

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading semantic model: %s", MODEL_NAME)
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
                logger.info("Semantic model loaded successfully")
    return _model


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    logger.info("Input truncated for semantic scoring: %s -> %s chars", len(text), max_chars)
    return text[:max_chars]


def validate_input(data):
    if not data:
        return "Request body is empty or not JSON."
    if not data.get('resume_text', '').strip():
        return "resume_text is required and cannot be empty."
    if not data.get('job_text', '').strip():
        return "job_text is required and cannot be empty."
    return None


@app.route('/score/tfidf', methods=['POST'])
def tfidf_score():
    data = request.get_json()
    error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    resume_text = preprocess(data['resume_text'])
    job_text = preprocess(data['job_text'])

    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        result = round(float(score) * 100, 2)
        logger.info(f"TF-IDF score computed: {result}")
        return jsonify({'tfidf_score': result})
    except Exception as e:
        logger.error(f"TF-IDF error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/score/semantic', methods=['POST'])
def semantic_score():
    data = request.get_json()
    error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    resume_text = clamp_text(preprocess(data['resume_text']), MAX_SEMANTIC_TEXT_CHARS)
    job_text = clamp_text(preprocess(data['job_text']), MAX_SEMANTIC_TEXT_CHARS)

    try:
        model = get_model()
        if torch is not None:
            with torch.inference_mode():
                embeddings = model.encode(
                    [resume_text, job_text],
                    convert_to_numpy=True,
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
        else:
            embeddings = model.encode(
                [resume_text, job_text],
                convert_to_numpy=True,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        score = float((embeddings[0] * embeddings[1]).sum())
        result = round(float(score) * 100, 2)
        logger.info(f"Semantic score computed: {result}")
        return jsonify({'semantic_score': result})
    except Exception as e:
        logger.error(f"Semantic error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL_NAME, 'model_loaded': _model is not None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)