from fastapi import FastAPI, File, UploadFile
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

MODEL_ID = "lambdalabs/pix2tex:4d559486cc0d69dc8b51236aa0f5bd8b7cbdeb18e718a22a42d9fa3bdca4ff2b"

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
