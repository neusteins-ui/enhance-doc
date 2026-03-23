"""
PDF Scan Enhancer
=================
Cleans and sharpens scanned PDFs without altering document content.

Usage:
    python enhance_pdf.py input.pdf
    python enhance_pdf.py input.pdf -o output.pdf
    python enhance_pdf.py input.pdf --dpi 400 --mode color
    python enhance_pdf.py input.pdf --mode bw        # Black & white (crisp for text docs)
    python enhance_pdf.py input.pdf --mode grayscale # Grayscale
    python enhance_pdf.py input.pdf --mode color     # Full color (default)
"""

import argparse
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import img2pdf
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Enhancement pipeline
# ---------------------------------------------------------------------------

def deskew(image_gray: np.ndarray) -> np.ndarray:
    """Detect and correct rotation using Hough line transform."""
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return image_gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return image_gray

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:  # skip tiny corrections
        return image_gray

    h, w = image_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image_gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_gray(img_gray: np.ndarray, sharpen_strength: float = 1.5) -> np.ndarray:
    """Full grayscale enhancement pipeline."""
    # 1. Denoise
    denoised = cv2.fastNlMeansDenoising(img_gray, None, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)

    # 2. Deskew
    deskewed = deskew(denoised)

    # 3. Adaptive contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(deskewed)

    # 4. Unsharp masking for sharpening
    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrasted, 1 + sharpen_strength,
                                blurred, -sharpen_strength, 0)

    return sharpened


def enhance_color(img_bgr: np.ndarray, sharpen_strength: float = 1.2) -> np.ndarray:
    """Full color enhancement pipeline."""
    # 1. Denoise (color)
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None,
                                                h=10, hColor=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

    # 2. Deskew (operate on gray for angle detection, apply to color)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                              minLineLength=100, maxLineGap=10)
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(a) < 45:
                    angles.append(a)
        if angles:
            angle = np.median(angles)

    if abs(angle) >= 0.3:
        h, w = denoised.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        denoised = cv2.warpAffine(denoised, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    # 3. CLAHE on L channel (LAB colorspace)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Unsharp masking
    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrasted, 1 + sharpen_strength,
                                blurred, -sharpen_strength, 0)

    return sharpened


def to_black_white(img_gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold for crisp black-and-white output."""
    denoised = cv2.fastNlMeansDenoising(img_gray, None, h=10)
    deskewed = deskew(denoised)
    bw = cv2.adaptiveThreshold(deskewed, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    # Light morphological cleanup (remove tiny speckles)
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return bw


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pdf(input_path: Path, output_path: Path, dpi: int, mode: str) -> None:
    print(f"Opening: {input_path}")
    doc = fitz.open(str(input_path))
    page_count = len(doc)
    print(f"Pages: {page_count}  |  DPI: {dpi}  |  Mode: {mode}")

    enhanced_images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is PDF base DPI

    for i, page in enumerate(doc):
        print(f"  Processing page {i + 1}/{page_count}...", end="\r")
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, pix.n)

        if mode == "bw":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            result = to_black_white(gray)
            pil_img = Image.fromarray(result).convert("L")

        elif mode == "grayscale":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            result = enhance_gray(gray)
            pil_img = Image.fromarray(result).convert("L")

        else:  # color
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result = enhance_color(bgr)
            rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

        enhanced_images.append(pil_img)

    doc.close()
    print(f"\nSaving enhanced PDF to: {output_path}")

    # Convert PIL images → PDF via img2pdf for lossless/high-quality output
    img_bytes_list = []
    for pil_img in enhanced_images:
        from io import BytesIO
        buf = BytesIO()
        if mode == "bw":
            pil_img.save(buf, format="PNG")
        else:
            pil_img.save(buf, format="JPEG", quality=95, dpi=(dpi, dpi))
        img_bytes_list.append(buf.getvalue())

    pdf_bytes = img2pdf.convert(img_bytes_list,
                                 layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi)))
    output_path.write_bytes(pdf_bytes)
    size_kb = output_path.stat().st_size / 1024
    print(f"Done. Output size: {size_kb:.0f} KB  ({output_path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhance scanned PDFs — denoise, deskew, sharpen, and boost contrast."
    )
    parser.add_argument("input", help="Path to the scanned PDF")
    parser.add_argument("-o", "--output",
                        help="Output path (default: <input>_enhanced.pdf)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI (default: 300; use 400-600 for very fine text)")
    parser.add_argument("--mode", choices=["color", "grayscale", "bw"], default="color",
                        help="Output color mode (default: color)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_enhanced")

    process_pdf(input_path, output_path, args.dpi, args.mode)


if __name__ == "__main__":
    main()
