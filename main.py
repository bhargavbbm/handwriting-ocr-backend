from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import replicate
import base64

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "lherman-cs/pix2tex:3a9817a9564f962d08d6579b0ce3f4bfefd4dc2621e073db1d3cb21915de7c2c"

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_base64 = base64.b64encode(image_bytes).decode()

    try:
        output = replicate.run(
            MODEL_ID,
            input={"image": f"data:image/png;base64,{img_base64}"}
        )
        return {"latex": output}
    except Exception as e:
        return {"error": str(e)}
