import requests
import os

MODEL_PATH = "model.pkl"

def download_model():
    MODEL_URL = os.getenv("MODEL_URL")

    if MODEL_URL is None:
        raise ValueError("MODEL_URL belum diset di Railway!")

    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("Downloading model from Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("Model downloaded:", MODEL_PATH)
