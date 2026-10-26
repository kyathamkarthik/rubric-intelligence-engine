"""
NLP Module computing cosine similarity between student reports and explicit rubric criteria.
"""

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Define explicit, human-readable rubric anchor concepts
RUBRIC_ANCHORS = {
    "rubric_arch_alignment": "Clear system architecture diagram, trade-off analysis, modular microservices, and design patterns.",
    "rubric_testing_alignment": "Comprehensive unit test suite, edge-case validation, CI/CD pipeline, and error handling coverage.",
    "rubric_eval_alignment": "Quantitative benchmark results, ablation study, dataset validation, and explainability analysis."
}

# Lazy-load the model to avoid blocking Streamlit startup
_model = None

def _get_model():
    global _model
    if _model is None and HAS_SBERT:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def compute_rubric_alignment(student_text: str) -> dict[str, float]:
    """
    Embeds the student text and computes cosine similarity against the rubric anchors.
    Returns bounded scores between 0.0 and 1.0.
    """
    if not student_text.strip():
        return {key: 0.0 for key in RUBRIC_ANCHORS.keys()}

    model = _get_model()
    
    if not model:
        # Graceful offline fallback
        return {key: 0.50 for key in RUBRIC_ANCHORS.keys()}

    # Compute embeddings
    student_emb = model.encode(student_text, convert_to_tensor=True)
    anchor_keys = list(RUBRIC_ANCHORS.keys())
    anchor_texts = list(RUBRIC_ANCHORS.values())
    anchor_embs = model.encode(anchor_texts, convert_to_tensor=True)

    # Compute cosine similarities and clamp to [0, 1]
    cosine_scores = util.cos_sim(student_emb, anchor_embs)[0].cpu().numpy()
    cosine_scores = np.clip(cosine_scores, 0.0, 1.0)

    return {key: round(float(score), 3) for key, score in zip(anchor_keys, cosine_scores)}