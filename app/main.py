from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.ocr_engine import process_pdf_file, process_image_file

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    content = await file.read()
    blocks = process_pdf_file(content)
    return {"blocks": blocks}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    content = await file.read()
    blocks = process_image_file(content)
    return {"blocks": blocks}

@app.get("/")
def home():
    return {"message": "Handwriting OCR backend running!"}
