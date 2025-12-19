from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from latexocr import LatexOCR
from PIL import Image
import io

app = FastAPI()

# enable frontend → backend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = LatexOCR()

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    try:
        latex = model(img)
        return {"latex": latex}
    except Exception as e:
        return {"error": str(e)}
