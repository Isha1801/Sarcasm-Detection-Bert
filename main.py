from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import requests
import re

app = FastAPI()

# Static files (CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# HTML templates
templates = Jinja2Templates(directory="templates")

# ========== 🔽 Model Download & Load ==========
MODEL_DIR = "model"
MODEL_FILE = "tf_model.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Google Drive share URL
SHARE_URL = "https://drive.google.com/file/d/15cLCBxGKXVhkYQQA70Pj5eZC_ozvjsCJ/view?usp=sharing"

def extract_drive_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def get_direct_download_url(SHARE_URL):
    file_id = extract_drive_id(SHARE_URL)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = get_direct_download_url(SHARE_URL)
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")

# Download if not present
download_model()

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("model")

model.load_weights(MODEL_PATH)
MAX_LEN = 64

# ========== 🔍 Prediction Helper ==========
def preprocess(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="tf")
    return inputs

# ========== 📄 Routes ==========
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    inputs = preprocess(text)
    outputs = model(inputs, training=False).logits
    prob = tf.nn.softmax(outputs, axis=1)
    prediction = np.argmax(prob)

    result = "Sarcastic 😏" if prediction == 1 else "Not Sarcastic 🙂"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "input_text": text
    })
