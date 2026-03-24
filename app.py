"""
PDF & Image Enhancer Dashboard — Flask server
Supports input:  PDF, JPG, JPEG, PNG, TIFF, BMP, WEBP
Supports output: PDF, JPG, PNG, TIFF, ZIP (multi-page images)
"""

import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import fitz
import img2pdf
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, url_for
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# Use /tmp on Vercel (read-only FS except /tmp); fall back to local dirs otherwise
import tempfile as _tempfile
_TMP = Path(_tempfile.gettempdir())
UPLOAD_DIR = _TMP / "enhance_uploads"
OUTPUT_DIR = _TMP / "enhance_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
ALL_EXTS   = PDF_EXTS | IMAGE_EXTS
OUT_FORMATS = {"pdf", "jpg", "png", "tiff"}


# ---------------------------------------------------------------------------
# Enhancement pipeline
# ---------------------------------------------------------------------------

def remove_shadow(img: np.ndarray) -> np.ndarray:
    """Remove uneven illumination, shadows, and dark phone-camera borders.
    Uses aggressive dilation + very large blur to estimate the background
    illumination map, then divides to normalize everything to ~white.
    """
    is_color = img.ndim == 3
    channels = cv2.split(img) if is_color else [img]
    # Large kernel with 2 iterations fills text gaps up to ~42px
    kernel = np.ones((21, 21), np.uint8)
    out = []
    for ch in channels:
        dilated = cv2.dilate(ch, kernel, iterations=2)
        # Very large sigma handles gradual shadows AND dark borders from phone scans
        bg = cv2.GaussianBlur(dilated, (0, 0), sigmaX=120)
        norm = cv2.divide(ch.astype(np.float32), bg.astype(np.float32), scale=255.0)
        out.append(np.clip(norm, 0, 255).astype(np.uint8))
    return cv2.merge(out) if is_color else out[0]


def auto_levels(img: np.ndarray) -> np.ndarray:
    """Stretch dynamic range so text → near-black and paper → pure white.
    Replaces separate whiten_background + darken_text with one clean operation.
    Works in LAB space for color images to preserve hue.
    """
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _auto_levels_ch(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return _auto_levels_ch(img)


def _auto_levels_ch(ch: np.ndarray) -> np.ndarray:
    """Auto-levels on a single channel: stretch, white-floor, darken text."""
    f = ch.astype(np.float32)
    # Find actual content range (ignore outlier pixels)
    p_dark = np.percentile(f, 2)     # ~text ink level
    p_light = np.percentile(f, 92)   # ~paper level
    if p_light <= p_dark + 10:
        return ch
    # Stretch so text → near 0 and paper → 255
    stretched = (f - p_dark) / (p_light - p_dark) * 255.0
    stretched = np.clip(stretched, 0, 255)
    # White floor: anything above 235 → pure white (kills residual gray on paper)
    stretched[stretched > 235] = 255
    # Darken text: adaptive gamma pulls dark pixels darker, leaves white alone
    norm = stretched / 255.0
    gamma = 1.0 + 0.6 * (1.0 - norm)  # 1.6 for black → 1.0 for white
    result = np.power(np.clip(norm, 1e-6, 1.0), gamma) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def deskew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
               for l in lines if l[0][2] != l[0][0]
               and abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 45]
    if not angles:
        return gray
    angle = np.median(angles)
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew_color(bgr: np.ndarray) -> np.ndarray:
    """Deskew a color image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return bgr
    angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))
              for l in lines if l[0][2] != l[0][0]
              and abs(np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))) < 45]
    if not angles:
        return bgr
    angle = np.median(angles)
    if abs(angle) < 0.3:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_color(bgr: np.ndarray) -> np.ndarray:
    """Color document enhancement — scanner-quality output from phone scans."""
    # 1. Bilateral filter — smooths noise while PRESERVING text edges
    #    (unlike fastNlMeans which smears everything including text)
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    # 2. Aggressive shadow removal — kills uneven lighting AND dark borders
    clean = remove_shadow(smooth)
    # 3. Auto-levels — stretches range, whitens paper, darkens text in one pass
    clean = auto_levels(clean)
    # 4. Deskew
    clean = deskew_color(clean)
    # 5. Tight sharpening — small sigma (0.7) = crisp edges without halos
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=0.7)
    return cv2.addWeighted(clean, 1.8, blurred, -0.8, 0)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Grayscale document enhancement."""
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clean = remove_shadow(smooth)
    clean = auto_levels(clean)
    clean = deskew(clean)
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=0.7)
    return cv2.addWeighted(clean, 1.8, blurred, -0.8, 0)


def to_bw(gray: np.ndarray) -> np.ndarray:
    """Black & white: maximum text clarity, pure white background."""
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clean = remove_shadow(smooth)
    clean = auto_levels(clean)
    bw = cv2.adaptiveThreshold(deskew(clean), 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def rotate_image(pil_img: Image.Image, degrees: int) -> Image.Image:
    """Rotate a PIL image by 0/90/180/270 degrees clockwise."""
    degrees = degrees % 360
    if degrees == 90:
        return pil_img.transpose(Image.ROTATE_270)
    elif degrees == 180:
        return pil_img.transpose(Image.ROTATE_180)
    elif degrees == 270:
        return pil_img.transpose(Image.ROTATE_90)
    return pil_img


def enhance_page(arr_rgb: np.ndarray, mode: str) -> Image.Image:
    """Enhance one RGB numpy array and return a PIL Image."""
    if mode == "bw":
        result = to_bw(cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY))
        return Image.fromarray(result).convert("L")
    elif mode == "grayscale":
        result = enhance_gray(cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY))
        return Image.fromarray(result).convert("L")
    else:
        result = enhance_color(cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Input → list of PIL images
# ---------------------------------------------------------------------------

def load_pages(input_path: Path, dpi: int) -> list[Image.Image]:
    """Load all pages from a PDF or image file as RGB PIL Images."""
    ext = input_path.suffix.lower()
    if ext in PDF_EXTS:
        doc = fitz.open(str(input_path))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            pages.append(Image.fromarray(arr).convert("RGB"))
        doc.close()
        return pages
    else:
        return [Image.open(str(input_path)).convert("RGB")]


# ---------------------------------------------------------------------------
# Enhanced PIL images → output bytes
# ---------------------------------------------------------------------------

def pages_to_pdf(pil_pages: list[Image.Image], dpi: int) -> bytes:
    chunks = []
    for pil in pil_pages:
        buf = BytesIO()
        # PNG for L-mode (BW/gray), JPEG for color
        if pil.mode == "L":
            pil.save(buf, format="PNG")
        else:
            pil.save(buf, format="JPEG", quality=95, dpi=(dpi, dpi))
        chunks.append(buf.getvalue())
    return img2pdf.convert(chunks, layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi)))


def pages_to_image_bytes(pil: Image.Image, fmt: str, dpi: int) -> bytes:
    """Serialize a single PIL image to the requested format."""
    buf = BytesIO()
    if fmt == "jpg":
        pil = pil.convert("RGB")  # JPEG can't be L or RGBA
        pil.save(buf, format="JPEG", quality=95, dpi=(dpi, dpi))
    elif fmt == "png":
        pil.save(buf, format="PNG")
    elif fmt == "tiff":
        pil.save(buf, format="TIFF", dpi=(dpi, dpi))
    return buf.getvalue()


def pages_to_zip(pil_pages: list[Image.Image], stem: str, fmt: str, dpi: int) -> bytes:
    """Pack multiple pages into a ZIP of images."""
    buf = BytesIO()
    ext = fmt  # jpg / png / tiff
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, pil in enumerate(pil_pages, 1):
            img_bytes = pages_to_image_bytes(pil, fmt, dpi)
            zf.writestr(f"{stem}_page{i:03d}.{ext}", img_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

def process(input_path: Path, mode: str, dpi: int,
            out_format: str, stem: str, rotation: int = 0) -> tuple[bytes, str]:
    """
    Returns (output_bytes, output_filename).
    out_format: 'pdf' | 'jpg' | 'png' | 'tiff'
    rotation:   0, 90, 180, 270 degrees clockwise
    """
    raw_pages  = load_pages(input_path, dpi)
    # Apply rotation before enhancement
    if rotation % 360 != 0:
        raw_pages = [rotate_image(p, rotation) for p in raw_pages]
    enh_pages  = [enhance_page(np.array(p), mode) for p in raw_pages]
    multi_page = len(enh_pages) > 1

    if out_format == "pdf":
        return pages_to_pdf(enh_pages, dpi), f"{stem}_enhanced.pdf"

    # Single image formats
    if multi_page:
        # ZIP up all pages
        return pages_to_zip(enh_pages, stem, out_format, dpi), f"{stem}_enhanced.zip"
    else:
        img_bytes = pages_to_image_bytes(enh_pages[0], out_format, dpi)
        return img_bytes, f"{stem}_enhanced.{out_format}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALL_EXTS:
        return jsonify(error=f"Unsupported type '{ext}'. Use: PDF, JPG, PNG, TIFF, BMP, WEBP"), 400

    mode       = request.form.get("mode", "color")
    out_format = request.form.get("out_format", "pdf")
    if out_format not in OUT_FORMATS:
        out_format = "pdf"
    dpi = int(request.form.get("dpi", 300))
    if dpi not in (150, 200, 300, 400, 600):
        dpi = 300
    rotation = int(request.form.get("rotation", 0)) % 360

    uid  = uuid.uuid4().hex[:8]
    stem = Path(f.filename).stem
    input_path = UPLOAD_DIR / f"{uid}_input{ext}"
    f.save(str(input_path))

    try:
        out_bytes, output_name = process(input_path, mode, dpi, out_format, stem, rotation)
    except Exception as e:
        input_path.unlink(missing_ok=True)
        return jsonify(error=str(e)), 500

    output_path = OUTPUT_DIR / f"{uid}_{output_name}"
    output_path.write_bytes(out_bytes)
    input_path.unlink(missing_ok=True)

    return jsonify(
        uid=uid,
        download_url=url_for("download", uid=uid, filename=output_name),
        preview_url=url_for("preview", uid=uid, filename=output_name),
        filename=output_name,
        size_kb=round(len(out_bytes) / 1024, 1),
    )


@app.route("/download/<uid>/<filename>")
def download(uid, filename):
    path = OUTPUT_DIR / f"{uid}_{filename}"
    if not path.exists():
        return "File not found", 404
    return send_file(str(path), as_attachment=True, download_name=filename)


@app.route("/preview-original", methods=["POST"])
def preview_original():
    """Convert an uploaded file's first page/frame to PNG for browser preview.
    Only needed for formats browsers can't display natively (TIFF).
    """
    if "file" not in request.files:
        return "No file", 400
    f = request.files["file"]
    raw = f.read()
    try:
        ext = Path(f.filename).suffix.lower()
        if ext in PDF_EXTS:
            doc = fitz.open(stream=raw, filetype="pdf")
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        else:
            pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return str(e), 500


@app.route("/preview/<uid>/<filename>")
def preview(uid, filename):
    """Serve a file inline for in-browser preview.
    TIFF and ZIP-page-1 are converted to PNG on the fly.
    """
    path = OUTPUT_DIR / f"{uid}_{filename}"
    if not path.exists():
        return "File not found", 404

    ext = Path(filename).suffix.lower()

    # PDF — render first page to PNG (iframe PDF blobs are unreliable across browsers)
    if ext == ".pdf":
        doc = fitz.open(str(path))
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    # TIFF — convert first page to PNG for browser display
    if ext in (".tiff", ".tif"):
        pil = Image.open(str(path)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    # ZIP — extract and serve the first image page as PNG
    if ext == ".zip":
        with zipfile.ZipFile(str(path)) as zf:
            names = sorted(zf.namelist())
            if not names:
                return "Empty archive", 404
            raw = zf.read(names[0])
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    # JPG / PNG — serve directly
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return send_file(str(path), mimetype=mime)


@app.route("/reprocess", methods=["POST"])
def reprocess():
    """Re-enhance an already-processed output with (optionally different) settings."""
    data       = request.get_json(force=True)
    src_uid    = data.get("uid", "")
    src_name   = data.get("filename", "")
    mode       = data.get("mode", "color")
    out_format = data.get("out_format", "pdf")
    dpi        = int(data.get("dpi", 300))

    if out_format not in OUT_FORMATS:
        out_format = "pdf"
    if dpi not in (150, 200, 300, 400, 600):
        dpi = 300

    src_path = OUTPUT_DIR / f"{src_uid}_{src_name}"
    if not src_path.exists():
        return jsonify(error="Source file not found — please re-upload"), 404

    # Strip any trailing _enhanced suffixes from stem for clean naming
    stem = Path(src_name).stem
    while stem.endswith("_enhanced"):
        stem = stem[: -len("_enhanced")]

    new_uid = uuid.uuid4().hex[:8]
    try:
        out_bytes, output_name = process(src_path, mode, dpi, out_format, stem)
    except Exception as e:
        return jsonify(error=str(e)), 500

    output_path = OUTPUT_DIR / f"{new_uid}_{output_name}"
    output_path.write_bytes(out_bytes)

    return jsonify(
        uid=new_uid,
        download_url=url_for("download", uid=new_uid, filename=output_name),
        preview_url=url_for("preview", uid=new_uid, filename=output_name),
        filename=output_name,
        size_kb=round(len(out_bytes) / 1024, 1),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
