#!/usr/bin/env python3
"""Florence-2 blue box detection with numbered bounding boxes and a results panel.

Install (Python 3.10+; use the same environment as the original OCR script):
    python -m pip install -U torch 'transformers>=4.57' pillow
Run:
    python florence2_ocr.py input.jpg -o result.png --json result.json
    python florence2_ocr.py input.jpg --device cpu
    python florence2_ocr.py input.jpg --text "A blue plastic box."

The default is the Transformers-native converted Florence-2 checkpoint.
Coordinates are pixels in the EXIF-oriented input image (origin: top left).
Bbox format: [xmin, ymin, xmax, ymax]. All returned candidates are retained.
Detected means at least one valid model candidate, not verified presence.
No OCR, tracking, confidence threshold, or independent color check is performed.
The input is kept in RGB because the prompt specifies color.
Source: https://huggingface.co/docs/transformers/model_doc/florence2
"""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
DEFAULT_TEXT = "A blue box."
DEFAULT_MODEL = "florence-community/Florence-2-large"


def load_font(path, size):
    """Use --font to select a font supporting the recognized language."""
    if path:
        return ImageFont.truetype(str(path), size)
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def wrap_line(text, font, max_width):
    """Wrap by measured pixel width, including long words and non-Latin text."""
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for character in paragraph:
            if current and font.getlength(current + character) > max_width:
                lines.append(current)
                current = character
            else:
                current += character
        lines.append(current)
    return lines


def make_regions(result, image_size):
    if not isinstance(result, dict) or "bboxes" not in result or "labels" not in result:
        raise ValueError("Grounding result must contain bboxes and labels.")
    labels = result["labels"]
    boxes = result["bboxes"]
    if len(labels) != len(boxes):
        raise ValueError("Grounding labels and bboxes have different lengths.")
    width, height = image_size
    regions = []
    for index, (label, box) in enumerate(zip(labels, boxes), 1):
        coordinates = [float(value) for value in box]
        if len(coordinates) != 4 or not all(map(math.isfinite, coordinates)):
            raise ValueError(f"Invalid bbox for region {index}: {box}")
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox ordering for region {index}: {box}")
        coordinates = [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                       max(0.0, min(x2, width)), max(0.0, min(y2, height))]
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            raise ValueError(f"Bbox outside input image for region {index}: {box}")
        regions.append({"id": index, "label": str(label), "bbox": coordinates})
    return regions


def render_result(image, regions, font_path=None, font_size=20, text=DEFAULT_TEXT):
    font = load_font(font_path, font_size)
    width = max(image.width, 800)
    margin = 20
    line_height = font_size + 10
    lines = [
        "Florence-2 box detection | Coordinates: pixels, origin = top left",
        *wrap_line("Target: " + text, font, width - 2 * margin),
        f"Detected: {bool(regions)} (model candidates)",
        f"Input: {image.width} x {image.height} | Regions: {len(regions)}",
        "BBox: [xmin, ymin, xmax, ymax]",
        "",
    ]
    for region in regions:
        lines.extend(wrap_line(
            f'[{region["id"]}] {region["label"]}', font, width - 2 * margin
        ))
        box = region["bbox"]
        coordinates = ", ".join(
            f"{value:.1f}" for value in box
        )
        lines.extend(wrap_line("BBox: [" + coordinates + "]", font, width - 2 * margin))
        lines.append("")
    if not regions:
        lines.append("No target box detected.")

    # Keep the original image unscaled, so its pixel coordinates remain valid.
    output = Image.new("RGB", (width, image.height + 2 * margin
                               + line_height * len(lines)), "white")
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    colors = ("#d62828", "#0077b6", "#008744", "#8e44ad", "#a65e00")
    for region in regions:
        color = colors[(region["id"] - 1) % len(colors)]
        box = region["bbox"]
        draw.rectangle(box, outline=color, width=3)
        label = str(region["id"])
        bounds = draw.textbbox((0, 0), label, font=font)
        label_w = bounds[2] - bounds[0] + 10
        label_h = bounds[3] - bounds[1] + 8
        x = max(0, min(min(box[0::2]), image.width - label_w))
        y = max(0, min(min(box[1::2]) - label_h, image.height - label_h))
        draw.rectangle((x, y, x + label_w, y + label_h), fill=color)
        draw.text((x + 5 - bounds[0], y + 4 - bounds[1]), label,
                  font=font, fill="white")

    draw.line((0, image.height, width, image.height), fill="#aaaaaa", width=2)
    y = image.height + margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill="#202020")
        y += line_height
    return output


def infer(image, args):
    # Lazy imports allow --help and rendering checks without model installation.
    try:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies: python -m pip install -U torch "
            "'transformers>=4.57' pillow"
        ) from exc
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Florence2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device).eval()
    inputs = processor(text=TASK + args.text, images=image, return_tensors="pt")
    inputs = {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point()
        else value.to(device=device)
        for key, value in inputs.items()
    }
    print(f"Detecting: {args.text} ...", file=sys.stderr)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             num_beams=3, do_sample=False)
    if ids.shape[-1] >= args.max_new_tokens:
        print("Warning: output may be truncated; increase --max-new-tokens "
              "or crop the input into smaller images.", file=sys.stderr)
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=TASK, image_size=image.size)
    if TASK not in parsed:
        raise RuntimeError("The model output did not contain a grounding result.")
    return make_regions(parsed[TASK], image.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Transformers-native Florence-2 model ID or local directory")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="English description of the target (default: A blue box.)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--font", type=Path, help="TTF/OTF/TTC font for result text")
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--json", type=Path, help="Optional machine-readable result file")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    args.text = args.text.strip()
    if args.font_size < 8 or args.max_new_tokens < 1:
        parser.error("--font-size must be >= 8; --max-new-tokens must be positive")
    output = args.output or args.image.with_name(args.image.stem + "_blue_box.png")
    if output.suffix.lower() != ".png":
        parser.error("The output image must have a .png extension")
    paths = [args.image.resolve(), output.resolve()]
    if args.json:
        paths.append(args.json.resolve())
    if len(paths) != len(set(paths)):
        parser.error("Input, output image, and optional JSON must have different paths")
    try:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        regions = infer(image, args)
        annotated = render_result(image, regions, args.font, args.font_size, args.text)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output, format="PNG")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({
                "task": TASK, "text_input": args.text, "model": args.model,
                "detected": bool(regions), "count": len(regions),
                "image_size": list(image.size),
                "coordinate_system": "EXIF-oriented image pixels; top-left origin",
                "regions": regions,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"detected": bool(regions), "count": len(regions),
                          "regions": regions}, ensure_ascii=False))
        print(f"Saved: {output.resolve()}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
