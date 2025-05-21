FROM python:3.10-slim

# 1. Create a non-root user
RUN useradd -m -u 1000 user

# 2. Set environment and working directory (as root)
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# 3. Install ffmpeg and pre-create folders with correct ownership
RUN apt-get update && apt-get install -y ffmpeg && \
    mkdir -p static audio_sample output cache/hf/transformers cache/whisper && \
    chown -R 1000:1000 static audio_sample output cache

# 4. Switch to non-root user for security
USER user

# 5. Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# 6. Copy app source code
COPY --chown=user . .

# 7. Expose Hugging Face's required port and start the server
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

