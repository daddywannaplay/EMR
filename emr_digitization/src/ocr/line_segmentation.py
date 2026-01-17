"""
Line segmentation utility for whole-page OCR.

Usage:
  - Import `detect_text_lines(image_path)` to get bounding boxes.
  - Run as a script to save cropped, resized line images for a page or a folder of pages.

The function uses simple projection/contour-based segmentation (binarize + dilate + contours)
which works well for horizontal text lines in prescriptions and lab reports.
"""
from typing import List, Tuple, Optional
import os
import cv2
import numpy as np


def detect_text_lines(image_path: str,
                      kernel_size: Tuple[int, int] = (30, 3),
                      min_area: int = 500,
                      padding: int = 4) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Detect horizontal text lines in a page image.

    Args:
        image_path: Path to the page image.
        kernel_size: Structuring element used for dilation (width, height).
        min_area: Minimum contour area to be considered a line.
        padding: Pixels to expand bbox on each side when cropping.

    Returns:
        boxes: List of bounding boxes (x, y, w, h) sorted top->bottom, left->right.
        gray: Grayscale image used for detection (uint8 ndarray).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Binarize (invert so text is white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to connect characters into line blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right for boxes on same line
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Optionally merge overlapping/close boxes on the same horizontal band
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        x, y, w, h = box
        # if vertically overlapping or close, merge horizontally
        if y <= py + ph + 10:
            nx = min(px, x)
            ny = min(py, y)
            nw = max(px + pw, x + w) - nx
            nh = max(py + ph, y + h) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    # Final boxes with padding (clamped to image size)
    h_img, w_img = gray.shape[:2]
    final = []
    for x, y, w, h in merged:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        final.append((x0, y0, x1 - x0, y1 - y0))

    return final, gray


def save_line_crops(image_path: str, out_dir: str, resize: Tuple[int, int] = (256, 64)) -> List[str]:
    """Detect lines in `image_path`, save cropped/resized lines to `out_dir`.

    Returns list of saved file paths in top->bottom order.
    """
    os.makedirs(out_dir, exist_ok=True)
    boxes, gray = detect_text_lines(image_path)
    saved = []
    base = os.path.splitext(os.path.basename(image_path))[0]
    for i, (x, y, w, h) in enumerate(boxes):
        crop = gray[y:y+h, x:x+w]
        try:
            crop_resized = cv2.resize(crop, resize)
        except Exception:
            # fallback: skip bad crops
            continue
        out_path = os.path.join(out_dir, f"{base}_line_{i}.png")
        cv2.imwrite(out_path, crop_resized)
        saved.append(out_path)
    return saved


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Detect text lines and save cropped line images')
    p.add_argument('input', help='Input image or folder')
    p.add_argument('-o', '--out', default='line_crops', help='Output folder')
    p.add_argument('--resize', type=int, nargs=2, default=(256, 64), help='Resize W H (default 256 64)')
    args = p.parse_args()

    inputs = []
    if os.path.isdir(args.input):
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                inputs.append(os.path.join(args.input, f))
    else:
        inputs = [args.input]

    for img_path in inputs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        outdir = os.path.join(args.out, stem)
        saved = save_line_crops(img_path, outdir, tuple(args.resize))
        print(f"{img_path} -> {len(saved)} lines saved to {outdir}")
