import requests
import os

MODEL_URL = os.getenv("MODEL")  # ambil dari Railway
MODEL_PATH = "model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        if MODEL_URL is None:
            raise ValueError("MODEL_URL belum diset di environment variable!")

        r = requests.get(MODEL, allow_redirects=True)

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

        print("Model downloaded.")
    else:
        print("Model already exists.")
