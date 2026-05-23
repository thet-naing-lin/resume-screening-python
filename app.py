from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.encode(["warmup"], show_progress_bar=False)  # warm up
logger.info("Model loaded and warmed up successfully.")


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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

    resume_text = preprocess(data['resume_text'])
    job_text = preprocess(data['job_text'])

    try:
        embeddings = model.encode(
            [resume_text, job_text],
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False
        )
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        result = round(float(score) * 100, 2)
        logger.info(f"Semantic score computed: {result}")
        return jsonify({'semantic_score': result})
    except Exception as e:
        logger.error(f"Semantic error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'all-MiniLM-L6-v2'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)