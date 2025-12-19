from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import replicate
import base64
import time

app = FastAPI()

# Allow Netlify frontend + others
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Working pix2tex model
MODEL = "yashgpt/pix2tex:8a1b5e294d7b8fc1bf2c3083a4a86745996457b0af0fa7f63c19e78146ee366b"


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """Convert handwritten physics to LaTeX using Replicate API."""
    try:
        # Read image to bytes
        image_bytes = await file.read()
        img_base64 = base64.b64encode(image_bytes).decode()

        # Try multiple times (Replicate rate limits often)
        retries = 3
        delay = 2

        for attempt in range(retries):
            try:
                output = replicate.run(
                    MODEL,
                    input={"image": f"data:image/png;base64,{img_base64}"}
                )

                # Handle weird Replicate null responses
                if not output:
                    raise ValueError("Model returned no output")

                return {"latex": output, "status": "success"}

            except replicate.exceptions.ReplicateError as e:
                if "rate limit" in str(e).lower():
                    # Wait then retry
                    time.sleep(delay)
                    delay *= 2
                    if attempt < retries - 1:
                        continue
                return {"error": "ReplicateError", "details": str(e)}

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return {"error": "Processing error", "details": str(e)}

    except Exception as e:
        return {"error": "Server error", "details": str(e)}
