import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pix2tex.cli import LatexOCR
from app.pdf_utils import pdf_to_images

# Initialize OCR models once at startup

# PaddleOCR for line detection
paddle = PaddleOCR(use_angle_cls=True, lang='en')

# TrOCR for handwritten text recognition
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# pix2tex for equation-to-LaTeX
pix_model = LatexOCR()


def is_math(text):
    """Simple classifier for math lines."""
    math_chars = "=+-/*^(){}[]><√∫ΣπωΩ"
    return any(c in text for c in math_chars)


def recognize_text_line(image_np):
    """Use TrOCR to read handwritten text."""
    pil_img = Image.fromarray(image_np)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def recognize_equation(image_np):
    """Convert handwritten equations to LaTeX."""
    pil_img = Image.fromarray(image_np)
    return pix_model(pil_img)


def process_image_np(img):
    """Full OCR pipeline for a single image."""
    results = paddle.ocr(img, cls=True)
    if not results or not results[0]:
        return []

    blocks = []

    for box, (text, conf) in results[0]:
        # Coordinates from PaddleOCR
        x1 = int(min(box[0][0], box[3][0]))
        y1 = int(min(box[0][1], box[1][1]))
        x2 = int(max(box[1][0], box[2][0]))
        y2 = int(max(box[2][1], box[3][1]))

        crop = img[y1:y2, x1:x2]

        if is_math(text):
            latex = recognize_equation(crop)
            blocks.append({"type": "equation", "content": latex})
        else:
            clean_text = recognize_text_line(crop)
            blocks.append({"type": "text", "content": clean_text})

    return blocks


def process_pdf_file(pdf_bytes):
    """Process a full PDF file."""
    pages = pdf_to_images(pdf_bytes)
    all_blocks = []

    for img_pil in pages:
        img_np = np.array(img_pil)
        blocks = process_image_np(img_np)
        all_blocks.extend(blocks)

    return all_blocks


def process_image_file(image_bytes):
    """Process a single image file."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    return process_image_np(img_np)
