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

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_base64 = base64.b64encode(image_bytes).decode()

    try:
        output = replicate.run(
            "luca-martial/pix2tex:latest",
            input={"image": f"data:image/png;base64,{img_base64}"}
        )
        return {"latex": output}
    except Exception as e:
        return {"error": str(e)}
