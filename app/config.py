import os
from dotenv import load_dotenv

# Load .env from project root (default behavior)
load_dotenv()

# Set cache directories to local folders (inside container or repo)
# These will be relative to the current working directory
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./cache/hf")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./cache/hf/transformers")
os.environ["WHISPER_CACHE"] = os.getenv("WHISPER_CACHE", "./cache/whisper")

# Optional: create cache directories at runtime
for env_var in ["HF_HOME", "TRANSFORMERS_CACHE", "WHISPER_CACHE"]:
    path = os.environ.get(env_var)
    if path and not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            print(f"⚠️ Warning: Cannot create {path}. Ensure it is writable by the app.")

# Hugging Face and LLM-related keys (should be set in .env or Hugging Face Secrets)
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")



