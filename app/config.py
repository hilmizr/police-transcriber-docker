"""
Global configuration & environment bootstrap.

Responsibilities
----------------
1. Load any `.env` file present at the project root (via python-dotenv).
2. Pin cache directories (HF, Transformers, Whisper) to local folders so the
   container or VM doesn’t redownload models on each run.
3. Expose all API / secret keys as environment variables—including the new ones
   for ElevenLabs Scribe.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# 1. Read .env (no-op if the file is absent)
# --------------------------------------------------------------------------- #
load_dotenv()

# --------------------------------------------------------------------------- #
# 2. Local cache paths (override via .env if you like)
# --------------------------------------------------------------------------- #
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./cache/hf")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./cache/hf/transformers")
os.environ["WHISPER_CACHE"] = os.getenv("WHISPER_CACHE", "./cache/whisper")

for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "WHISPER_CACHE"):
    path = Path(os.environ[env_var])
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"⚠️  Cannot create cache dir {path!s}. Make sure it’s writable.")

# --------------------------------------------------------------------------- #
# 3. Secrets & API keys
# --------------------------------------------------------------------------- #
# Existing keys
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

# >>> NEW for ElevenLabs Scribe <<<
os.environ["XI_API_KEY"] = os.getenv("XI_API_KEY", "")                   
os.environ["SCRIBE_ENDPOINT"] = os.getenv(                              
    "SCRIBE_ENDPOINT",
    "https://api.elevenlabs.io/v1/speech-to-text",
)
