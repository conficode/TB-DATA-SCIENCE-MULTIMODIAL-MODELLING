# Hugging Face Spaces (Docker SDK) / any container host
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# No-op when the LFS files are already present (HF Spaces); downloads them otherwise
RUN python scripts/fetch_models.py

EXPOSE 7860
CMD gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:${PORT}
