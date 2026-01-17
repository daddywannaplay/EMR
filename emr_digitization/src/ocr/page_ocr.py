"""
Page-level OCR pipeline: detect lines, run CRNN on each line, and merge into one text file.

Usage examples:
  python src/ocr/page_ocr.py page.jpg \
      --model ocr_models/prescriptions_best_model.pt \
      --vocab ocr_models/prescriptions_vocab.json \
      --out ocr_models/prescriptions_page_text

  python src/ocr/page_ocr.py pages_folder --out ocr_models/prescriptions_page_text

This script uses `line_segmentation.detect_text_lines` to find lines and the existing
`CRNN` model (from `train_model.py`) for per-line OCR (greedy CTC decoding).
"""
import os
import json
from typing import List, Tuple

import cv2
import torch
import numpy as np

from train_model import CRNN
from src.ocr.line_segmentation import detect_text_lines


def preprocess_line_image(img: np.ndarray) -> torch.Tensor:
    # img: grayscale numpy array
    img = cv2.resize(img, (256, 64))
    t = torch.FloatTensor(img.astype('float32') / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t


def greedy_ctc_decode(logits: torch.Tensor, idx2char: dict, blank_idx: int = 0) -> str:
    # logits: [T, B=1, C]
    preds = logits.argmax(dim=2).squeeze(1).cpu().numpy().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p == prev:
            prev = p
            continue
        if p == blank_idx:
            prev = p
            continue
        collapsed.append(p)
        prev = p
    return ''.join([idx2char.get(i, '') for i in collapsed])


def ocr_page(image_path: str, model_path: str, vocab_path: str, out_dir: str) -> Tuple[str, List[dict]]:
    os.makedirs(out_dir, exist_ok=True)

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    idx2char = {int(k): v for k, v in vocab.get('idx2char', {}).items()} if 'idx2char' in vocab else {}
    vocab_size = vocab.get('vocab_size', len(idx2char) or 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    boxes, gray = detect_text_lines(image_path)

    lines = []
    for (x, y, w, h) in boxes:
        crop = gray[y:y+h, x:x+w]
        inp = preprocess_line_image(crop).to(device)
        with torch.no_grad():
            logits = model(inp)  # [T, B, C]
        text = greedy_ctc_decode(logits, idx2char, blank_idx=0)
        lines.append({'box': (x, y, w, h), 'text': text})

    # sort lines top->bottom then left->right (boxes are expected sorted already)
    lines_sorted = sorted(lines, key=lambda r: (r['box'][1], r['box'][0]))

    page_text = '\n'.join([l['text'] for l in lines_sorted])

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(out_dir, base + '.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(page_text)

    results = [{'input_image': image_path, 'output_text_file': out_file, 'lines': lines_sorted}]
    return out_file, results


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Page OCR: segment lines, OCR with CRNN, merge to .txt')
    p.add_argument('input', help='Input image file or folder')
    p.add_argument('--model', required=True, help='Path to CRNN model .pt')
    p.add_argument('--vocab', required=True, help='Path to vocab json')
    p.add_argument('--out', default='page_texts', help='Output folder for page .txt files')
    args = p.parse_args()

    inputs = []
    if os.path.isdir(args.input):
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                inputs.append(os.path.join(args.input, f))
    else:
        inputs = [args.input]

    os.makedirs(args.out, exist_ok=True)
    for img_path in inputs:
        out_file, results = ocr_page(img_path, args.model, args.vocab, args.out)
        print(f"{img_path} -> {out_file} ({len(results[0]['lines'])} lines)")
