from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import replicate
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "zsxkib/ocr-math-handwriting:b4b6bed3bd2eb2de0c9b2e563e0fb5bf5e2bd6e4cc0f169fb4df75f1a4bb7a57"

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    img_bytes = await file.read()
    b64 = base64.b64encode(img_bytes).decode()

    try:
        result = replicate.run(
            MODEL_ID,
            input={"image": f"data:image/png;base64,{b64}"}
        )

        return {"latex": result}

    except Exception as e:
        return {"error": str(e)}
