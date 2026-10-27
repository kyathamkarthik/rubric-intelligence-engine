# Use an official Python slim runtime for a smaller footprint
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create a non-root user for security compliance
RUN addgroup --system appgroup && adduser --system --group appuser

# Set the working directory
WORKDIR /app

# Install system dependencies required for building ML packages and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies early to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-bake the Sentence-Transformer model weights directly into the image layer
# This prevents startup lag and allows the container to run in air-gapped environments
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Ensure data directories exist and transfer ownership to the non-root user
RUN mkdir -p /app/data/raw /app/data/feedback /app/.cache && \
    chown -R appuser:appgroup /app

# Drop root privileges
USER appuser

# Expose Streamlit's default port
EXPOSE 8501

# Container healthcheck using Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit Calibration Engine
CMD ["streamlit", "run", "app.py"]