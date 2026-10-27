# Rubric Intelligence Engine

A Python-based, rubric-aware project quality intelligence and mentor calibration engine built with Streamlit. The system evaluates project submissions using a combination of static code analysis, rubric-aligned NLP scoring, uncertainty estimation, and human-in-the-loop mentor calibration.

This project is designed for educational and assessment workflows where you want to:

- score student or team project submissions against rubric dimensions,
- detect uncertain predictions that need reviewer attention,
- compare AI predictions with human mentor grading,
- monitor calibration drift and reviewer bias over time.

## Overview

The application analyzes submission quality from multiple signals:

- Source code structure and complexity through AST-based analysis
- Documentation and code quality signals such as docstring coverage and lint-style warnings
- NLP alignment between project reports and rubric criteria
- Conformal prediction intervals for uncertainty-aware scoring
- Human mentor overrides and calibration monitoring

## Key Features

- Batch queue review and calibration workflow
- Live file upload and rubric scoring
- Mentor override and approval workflow
- Active learning retraining using mentor feedback
- Uncertainty-aware scoring using MAPIE conformal regressors
- SHAP-based explainability for model decisions
- Mentor drift analytics to detect harshness or leniency bias

## Project Architecture

```text
rubric-intelligence-engine/
├── app.py                              # Streamlit application entry point
├── requirements.txt                    # Python dependencies
├── scripts/
│   └── generate_data.py                # Synthetic dataset generation
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features/
│   │   ├── code_analyzer.py            # AST-based code quality metrics
│   │   └── rubric_aligner.py           # NLP alignment against rubric anchors
│   └── monitoring/
│       └── mentor_drift.py             # Mentor bias / drift analytics
├── data/
│   ├── raw/
│   └── feedback/
└── README.md
```

## How It Works

### 1. Code analysis
The project parses Python files using Python's AST module to extract:

- average cyclomatic complexity,
- docstring coverage,
- structural warning indicators.

### 2. Rubric alignment
Student reports are compared against rubric anchor concepts using sentence-transformer embeddings, producing alignment scores for:

- system architecture,
- testing quality,
- evaluation and benchmarking.

### 3. Prediction and uncertainty
The app trains Random Forest regressors for rubric targets and wraps them in MAPIE conformal regressors to produce prediction intervals. This helps flag uncertain cases for mentor review.

### 4. Calibration and feedback loop
Mentors can approve or override AI scores. Those decisions are saved to a CSV log and can be used to retrain the model, improving alignment over time.

### 5. Mentor drift monitoring
The monitoring module aggregates mentor offsets relative to AI predictions, helping identify bias patterns such as harshness or leniency.

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- MAPIE
- SHAP
- sentence-transformers
- pandas + NumPy
- matplotlib
- pingouin

## Installation

Clone the repository:

```bash
git clone https://github.com/kyathamkarthik/rubric-intelligence-engine.git
cd rubric-intelligence-engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will generate synthetic data automatically when needed, or you can generate it manually:

```bash
python scripts/generate_data.py
```

## Usage

After launching the app, you can:

1. Open the batch review tab to inspect pending submissions.
2. Review flagged submissions with model uncertainty and SHAP explanations.
3. Approve or override AI predictions using mentor calibration controls.
4. Upload live Python code and project reports for instant rubric-based scoring.
5. Monitor mentor grading drift and calibration health in the monitoring tab.

## Example Outputs

The application produces:

- predicted technical correctness and completeness scores,
- 90% confidence intervals,
- uncertainty-based review flags,
- mentor override logs,
- per-mentor bias summaries.

## Example Repository Workflow

```text
Generate data -> Train models -> Review flagged submissions -> Mentor override -> Retrain -> Monitor drift
```

## Notes

This repository uses synthetic data by default to demonstrate the scoring workflow and calibration pipeline. It is designed as a foundation for extending to real student submissions, institutional rubric schemas, and production-grade review systems.

## License

This project is currently provided without a specific license file. If you plan to distribute it publicly, consider adding an open-source license such as MIT or Apache 2.0.

## Contributing

Contributions are welcome. If you would like to improve the engine, consider:

- adding support for real rubric datasets,
- improving the feature extraction pipeline,
- expanding mentor analytics,
- integrating deployment or API endpoints.

## Contact

For questions or collaboration, reach out via the repository owner or the project maintainer.
