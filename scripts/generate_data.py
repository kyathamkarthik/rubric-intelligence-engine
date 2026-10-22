"""
Synthetic Multimodal Dataset Generator for the Rubric-Aware Project Quality Engine.
Generates code quality metrics, Sentence-Transformer text embeddings, and historical rubric scores.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Attempt to load SentenceTransformer; provide graceful fallback if offline
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


PROJECT_REPORT_TEMPLATES = [
    "Implemented a distributed microservice architecture with comprehensive unit tests, CI/CD pipelines, and edge-case error handling.",
    "Built a baseline CRUD web application. Core endpoints work as expected, though unit test coverage and concurrency locks are incomplete.",
    "Developed the machine learning pipeline with hyperparameter tuning, SHAP explainability, and modular data validation schemas.",
    "Submitted partial prototype. Several modules throw runtime exceptions on null inputs and documentation is minimal.",
    "Engineered an end-to-end streaming pipeline with high throughput, thorough integration tests, and detailed architectural tradeoff analysis.",
    "Basic script implementation with hardcoded file paths, high cyclomatic complexity, and no automated test suite.",
]


def compute_text_embeddings(texts: list[str], n_components: int = 3, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """
    Encodes project report texts using Sentence-Transformers (all-MiniLM-L6-v2)
    and extracts both the raw embedding JSON and PCA-compressed semantic dimensions for SHAP.
    """
    if HAS_SBERT:
        try:
            print("Loading Sentence-Transformer model ('all-MiniLM-L6-v2')...")
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            raw_embeddings = encoder.encode(texts, show_progress_bar=False)
        except Exception as exc:
            print(f"[Warning] Could not download SBERT model ({exc}). Using deterministic synthetic embeddings.")
            rng = np.random.default_rng(seed)
            raw_embeddings = rng.normal(0, 1, size=(len(texts), 384))
    else:
        print("[Warning] sentence-transformers not installed. Using deterministic synthetic embeddings.")
        rng = np.random.default_rng(seed)
        raw_embeddings = rng.normal(0, 1, size=(len(texts), 384))

    pca = PCA(n_components=n_components, random_state=seed)
    compressed_embeddings = pca.fit_transform(raw_embeddings)

    # Store truncated raw vector preview as JSON string for schema completeness
    raw_json_strings = [
        json.dumps(np.round(vec[:8], 4).tolist()) for vec in raw_embeddings
    ]
    return compressed_embeddings, raw_json_strings


def generate_synthetic_dataset(n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic student submissions with correlated rubric scores and edge cases."""
    rng = np.random.default_rng(seed)

    submission_ids = [f"SUB-{1000 + i}" for i in range(n_samples)]
    student_ids = [f"STU-{rng.integers(200, 999)}" for _ in range(n_samples)]

    # 1. Code Quality Metrics
    # Cyclomatic complexity score (1 = clean/modular, 25 = tangled/spaghetti code)
    code_complexity_score = np.round(rng.uniform(2.0, 25.0, size=n_samples), 2)
    # Automated test coverage ratio (0.0 to 1.0)
    test_coverage_ratio = np.round(rng.beta(a=2.5, b=1.8, size=n_samples), 3)
    # Docstring & architecture documentation coverage (0.0 to 1.0)
    docstring_coverage = np.round(rng.uniform(0.1, 1.0, size=n_samples), 3)
    # Static analysis lint warnings count
    lint_warning_count = rng.poisson(lam=8, size=n_samples)

    # 2. Text Reports & Embeddings
    report_texts = [
        PROJECT_REPORT_TEMPLATES[i % len(PROJECT_REPORT_TEMPLATES)]
        for i in range(n_samples)
    ]
    compressed_emb, raw_emb_json = compute_text_embeddings(report_texts, n_components=3, seed=seed)

    # 3. Synthesize Ground-Truth Historical Rubric Labels (0 - 100 scale)
    # Technical Correctness driven strongly by test coverage, low complexity, and low lint warnings
    tech_noise = rng.normal(0, 4.5, size=n_samples)
    technical_correctness = (
        45.0
        + 42.0 * test_coverage_ratio
        - 1.15 * code_complexity_score
        - 0.65 * lint_warning_count
        + 4.0 * compressed_emb[:, 0]
        + tech_noise
    )
    technical_correctness = np.clip(np.round(technical_correctness, 1), 0.0, 100.0)

    # Completeness driven by test coverage, docstring coverage, and semantic report depth
    comp_noise = rng.normal(0, 5.0, size=n_samples)
    completeness_score = (
        35.0
        + 35.0 * test_coverage_ratio
        + 28.0 * docstring_coverage
        - 0.60 * code_complexity_score
        + 5.5 * compressed_emb[:, 0]
        - 3.0 * compressed_emb[:, 1]
        + comp_noise
    )
    completeness_score = np.clip(np.round(completeness_score, 1), 0.0, 100.0)

    # 4. Inject Out-of-Distribution (OOD) / Ambiguous Submissions to trigger High Model Uncertainty
    # E.g., High test coverage but extremely high complexity & poor documentation
    ood_indices = rng.choice(n_samples, size=18, replace=False)
    for idx in ood_indices:
        code_complexity_score[idx] = np.round(rng.uniform(22.0, 30.0), 2)
        test_coverage_ratio[idx] = np.round(rng.uniform(0.85, 0.99), 3)
        docstring_coverage[idx] = np.round(rng.uniform(0.05, 0.20), 3)
        lint_warning_count[idx] = int(rng.integers(20, 40))

    df = pd.DataFrame({
        "submission_id": submission_ids,
        "student_id": student_ids,
        "dataset_split": ["historical_train" if i < 115 else "pending_evaluation" for i in range(n_samples)],
        "code_complexity_score": code_complexity_score,
        "test_coverage_ratio": test_coverage_ratio,
        "docstring_coverage": docstring_coverage,
        "lint_warning_count": lint_warning_count,
        "submission_text_embedding_0": np.round(compressed_emb[:, 0], 4),
        "submission_text_embedding_1": np.round(compressed_emb[:, 1], 4),
        "submission_text_embedding_2": np.round(compressed_emb[:, 2], 4),
        "submission_text_embedding": raw_emb_json,
        "report_summary": report_texts,
        "technical_correctness": technical_correctness,
        "completeness_score": completeness_score,
    })

    return df


if __name__ == "__main__":
    output_path = Path("data/raw/synthetic_submissions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_generated = generate_synthetic_dataset(n_samples=150)
    df_generated.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df_generated)} synthetic submissions at: {output_path.resolve()}")
    print(df_generated[["submission_id", "dataset_split", "code_complexity_score", "test_coverage_ratio", "technical_correctness", "completeness_score"]].head())