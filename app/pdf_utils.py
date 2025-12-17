import io
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(pdf_bytes):
    """
    Convert a PDF (bytes) into a list of PIL images.
    Each page becomes a separate image.
    """
    pdf = fitz.open("pdf", pdf_bytes)
    images = []

    for page in pdf:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images
