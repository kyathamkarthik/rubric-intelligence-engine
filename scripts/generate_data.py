"""
Updated Synthetic Dataset Generator for the Rubric-Aware Project Engine.
Now generates explicit Rubric-Aligned NLP features instead of raw PCA embeddings.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic submissions using the updated P3 Rubric NLP schema."""
    rng = np.random.default_rng(seed)

    submission_ids = [f"SUB-{1000 + i}" for i in range(n_samples)]
    student_ids = [f"STU-{rng.integers(200, 999)}" for _ in range(n_samples)]

    # 1. Code Quality Metrics
    code_complexity_score = np.round(rng.uniform(2.0, 25.0, size=n_samples), 2)
    test_coverage_ratio = np.round(rng.beta(a=2.5, b=1.8, size=n_samples), 3)
    docstring_coverage = np.round(rng.uniform(0.1, 1.0, size=n_samples), 3)
    lint_warning_count = rng.poisson(lam=8, size=n_samples)

    # 2. P3 Upgraded NLP Alignment Metrics (0.0 to 1.0 cosine similarity equivalents)
    # Simulate correlation: high test coverage usually implies high testing alignment in text
    rubric_arch_alignment = np.clip(np.round(rng.normal(0.65, 0.20, size=n_samples), 3), 0.0, 1.0)
    rubric_testing_alignment = np.clip(np.round(test_coverage_ratio + rng.normal(0, 0.15, size=n_samples), 3), 0.0, 1.0)
    rubric_eval_alignment = np.clip(np.round(rng.normal(0.50, 0.25, size=n_samples), 3), 0.0, 1.0)

    report_texts = [
        "Architecture utilizes microservices with comprehensive unit tests and evaluation benchmarks." 
        for _ in range(n_samples)
    ]

    # 3. Synthesize Ground-Truth Historical Rubric Labels (0 - 100 scale)
    # Tech Score relies heavily on tests, complexity, and the NLP testing alignment
    technical_correctness = (
        30.0
        + 30.0 * test_coverage_ratio
        - 1.0 * code_complexity_score
        - 0.5 * lint_warning_count
        + 35.0 * rubric_testing_alignment
        + rng.normal(0, 3.5, size=n_samples)
    )
    technical_correctness = np.clip(np.round(technical_correctness, 1), 0.0, 100.0)

    # Completeness relies on documentation and the NLP architecture/eval alignments
    completeness_score = (
        25.0
        + 25.0 * docstring_coverage
        + 25.0 * rubric_arch_alignment
        + 25.0 * rubric_eval_alignment
        + rng.normal(0, 4.0, size=n_samples)
    )
    completeness_score = np.clip(np.round(completeness_score, 1), 0.0, 100.0)

    # 4. Inject Out-of-Distribution (OOD) cases to trigger the MAPIE Conformal Uncertainty alerts
    ood_indices = rng.choice(n_samples, size=15, replace=False)
    for idx in ood_indices:
        code_complexity_score[idx] = np.round(rng.uniform(22.0, 30.0), 2)
        test_coverage_ratio[idx] = np.round(rng.uniform(0.85, 0.99), 3) # Conflicting: High coverage, awful complexity
        rubric_testing_alignment[idx] = 0.10  # Conflicting: Code says tests exist, report mentions none

    df = pd.DataFrame({
        "submission_id": submission_ids,
        "student_id": student_ids,
        "dataset_split": ["historical_train" if i < 115 else "pending_evaluation" for i in range(n_samples)],
        "code_complexity_score": code_complexity_score,
        "test_coverage_ratio": test_coverage_ratio,
        "docstring_coverage": docstring_coverage,
        "lint_warning_count": lint_warning_count,
        "rubric_arch_alignment": rubric_arch_alignment,
        "rubric_testing_alignment": rubric_testing_alignment,
        "rubric_eval_alignment": rubric_eval_alignment,
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
    print(f"✅ Generated {len(df_generated)} synthetic submissions mapped to explicit Rubric NLP anchors.")