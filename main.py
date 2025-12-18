import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64
from mfr import latex_ocr

app = FastAPI()

# Allow frontend (Netlify) to use backend (Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = latex_ocr.LatexOCR()   # load model once (fast!)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    img_bytes = await file.read()
    latex = model(img_bytes)   # run model
    return {"latex": latex}
