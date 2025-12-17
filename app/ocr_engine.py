import io
import numpy as np
from PIL import Image
import cv2

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pix2tex.cli import LatexOCR
from app.pdf_utils import pdf_to_images

# Load models
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

pix_model = LatexOCR()


def is_math(text):
    math_chars = "=+-/*^(){}[]><√∫ΣπωΩ"
    return any(c in text for c in math_chars)


def recognize_text_line(image_np):
    img = Image.fromarray(image_np)
    pixel = processor(images=img, return_tensors="pt").pixel_values
    ids = trocr_model.generate(pixel)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def recognize_equation(image_np):
    img = Image.fromarray(image_np)
    return pix_model(img)


def segment_lines(img_np):
    """Detect horizontal handwritten lines using OpenCV."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )

    # Dilate horizontally to form line blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h < 20 or w < 50:
            continue

        crop = img_np[y:y+h, x:x+w]
        lines.append(crop)

    # Sort lines from top to bottom
    lines = sorted(lines, key=lambda c: cv2.boundingRect(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY))[1])

    return lines


def process_image_np(img_np):
    lines = segment_lines(img_np)
    blocks = []

    for line in lines:
        # Quick check if likely mathematical: many symbols detected
        text_preview = recognize_text_line(line)

        if is_math(text_preview):
            latex = recognize_equation(line)
            blocks.append({"type": "equation", "content": latex})
        else:
            blocks.append({"type": "text", "content": text_preview})

    return blocks


def process_pdf_file(pdf_bytes):
    pages = pdf_to_images(pdf_bytes)
    all_blocks = []

    for img in pages:
        img_np = np.array(img)
        all_blocks.extend(process_image_np(img_np))

    return all_blocks


def process_image_file(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    return process_image_np(img_np)
