from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

app = FastAPI()

# Allow all origins (we can restrict later to your Netlify domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Replicate token (set in Render environment)
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not REPLICATE_TOKEN:
        return {"error": "Missing REPLICATE_API_TOKEN on server"}

    # Read uploaded image
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    # Replicate model endpoint
    url = "https://api.replicate.com/v1/models/lucataco/latexify-pytorch/predictions"

    headers = {
        "Authorization": f"Bearer {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {"input": {"image": "data:image/png;base64," + img_b64}}

    # Create prediction
    prediction = requests.post(url, json=payload, headers=headers).json()

    if "id" not in prediction:
        return {"error": prediction}

    prediction_id = prediction["id"]

    # Poll until done
    while True:
        result = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers=headers,
        ).json()

        status = result["status"]

        if status == "succeeded":
            return {"latex": result["output"]}

        if status == "failed":
            return {"error": "Prediction failed"}

        # Continue polling
