#!/usr/bin/env python3
"""Choose the largest Florence-2 bbox and compare its center to a vertical line.

使い方（Florence-2で再推論は行いません）:
  python florence2_phrase.py input.jpg -o boxes.png --json boxes.json
  python bbox_direction.py boxes.png --boxes-json boxes.json --line-x 620
  python bbox_direction.py boxes.png --boxes-json boxes.json

--line-x省略時は画像上をクリックしてロボット中心のX座標を指定します。
GUIにはUbuntuのpython3-tkが必要です。数値指定時はGUI不要です。
依存: pillow（元のFlorence-2スクリプトと同じ環境で実行可能）
画像は元スクリプトが出力する無拡大・無縮小のPNGを指定してください。
bbox座標は画像ではなく対応するJSONから読みます。左右は最大bboxの中心で判定。
通常の非鏡像カメラなら画像左=左回転、画像右=右回転です。
鏡像画像の場合は--mirroredを指定してください。ロボットへの指令送信はしません。
"""
import argparse
import json
import math
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def read_boxes(path, displayed_size):
    data = json.loads(path.read_text(encoding='utf-8'))
    size = data.get('image_size')
    if not isinstance(size, list) or len(size) != 2 or any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size
    ):
        raise ValueError('JSON image_size must be [width, height].')
    w, h = size
    if displayed_size[0] < w or displayed_size[1] < h:
        raise ValueError('画像がJSONの元画像より小さいです。リサイズしていない出力PNGを指定してください。')
    raw_regions = data.get('regions')
    if not isinstance(raw_regions, list):
        raise ValueError('JSON must contain a regions array with bbox coordinates.')
    regions = []
    for index, region in enumerate(raw_regions, 1):
        if not isinstance(region, dict):
            raise ValueError(f'Invalid region {index}.')
        box = region.get('bbox')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f'Region {index} needs bbox [xmin, ymin, xmax, ymax].')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in box):
            raise ValueError(f'Region {index} has invalid coordinates.')
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f'Region {index} bbox is outside the source image or has zero area.')
        regions.append({'id': region.get('id', index), 'label': str(region.get('label', 'box')),
                        'bbox': [x1, y1, x2, y2], 'area_px2': (x2-x1)*(y2-y1)})
    return size, regions


def evaluate(regions, line_x, deadband=0, mirrored=False):
    result = {'detected': bool(regions), 'candidate_count': len(regions),
              'line_x': line_x, 'deadband_px': deadband, 'mirrored': mirrored,
              'selected': None, 'side': 'not_detected', 'turn_direction': 'undetermined'}
    if not regions:
        return result
    # max keeps the first candidate when the areas are equal.
    selected = dict(max(regions, key=lambda r: r['area_px2']))
    x1, y1, x2, y2 = selected['bbox']
    cx, cy = (x1+x2)/2, (y1+y2)/2
    offset = cx-line_x
    side = 'left' if offset < -deadband else 'right' if offset > deadband else 'center'
    turn = side
    if mirrored and side in ('left', 'right'):
        turn = 'right' if side == 'left' else 'left'
    selected.update({'center_x': cx, 'center_y': cy, 'offset_x_px': offset,
                     'crosses_line': x1 <= line_x <= x2})
    result.update({'selected': selected, 'side': side, 'turn_direction': turn})
    return result


def pick_line(image, source_size):
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as exc:
        raise RuntimeError('GUIにはpython3-tkが必要です。sudo apt install python3-tk、または--line-xを指定してください。') from exc
    root = None
    try:
        root = tk.Tk()
        root.title('Click robot center line / Enter to confirm / Esc to cancel')
        w, h = source_size
        scale = min(1.0, (root.winfo_screenwidth()-100)/w,
                    (root.winfo_screenheight()-160)/h)
        dw, dh = max(1, int(w*scale)), max(1, int(h*scale))
        preview = image.crop((0, 0, w, h)).resize((dw, dh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        canvas = tk.Canvas(root, width=dw, height=dh, highlightthickness=0)
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=photo)
        label = tk.Label(root, text='Click the robot center, then press Enter or Confirm.')
        label.pack()
        choice = {'x': None, 'confirmed': False}
        def click(event):
            choice['x'] = max(0, min(w-1, event.x*w/dw))
            canvas.delete('center_line')
            canvas.create_line(event.x, 0, event.x, dh, fill='#ff00ff', width=3, tags='center_line')
            label.config(text=f"Robot center X = {choice['x']:.2f} px. Enter or Confirm to save.")
        def confirm(event=None):
            if choice['x'] is not None:
                choice['confirmed'] = True
                root.destroy()
        canvas.bind('<Button-1>', click)
        root.bind('<Return>', confirm)
        root.bind('<Escape>', lambda event: root.destroy())
        tk.Button(root, text='Confirm', command=confirm).pack()
        root.mainloop()
        if not choice['confirmed']:
            raise RuntimeError('中心線の選択をキャンセルしました。')
        return choice['x']
    except tk.TclError as exc:
        if root is not None:
            try: root.destroy()
            except tk.TclError: pass
        raise RuntimeError('GUIを開けません。画面のある環境で実行するか、--line-xで座標を指定してください。') from exc


def draw_result(image, size, result):
    w, h = size
    output = image.copy()
    draw = ImageDraw.Draw(output)
    x = result['line_x']
    draw.line((x, 0, x, h-1), fill='#ff00ff', width=4)
    if result['selected']:
        box = result['selected']['bbox']
        draw.rectangle(box, outline='#ffcc00', width=5)
        cx, cy = result['selected']['center_x'], result['selected']['center_y']
        draw.line((x, cy, cx, cy), fill='#ffcc00', width=3)
        draw.ellipse((cx-6, cy-6, cx+6, cy+6), fill='#ffcc00', outline='black')
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    lines = [f"Robot center X: {x:.2f} px | Candidates: {result['candidate_count']}",
             f"Side: {result['side']} | Turn: {result['turn_direction']}"]
    if result['selected']:
        s = result['selected']
        lines += [f"Selected ID: {s['id']} | Area: {s['area_px2']:.1f} px2",
                  f"BBox center X: {s['center_x']:.2f} | Offset X: {s['offset_x_px']:+.2f} px"]
    else:
        lines += ['No bbox detected: turn direction is undetermined.']
    # Add a new panel below the existing Florence output, preserving its coordinates.
    panel = Image.new('RGB', (max(output.width, 800), output.height+30*len(lines)+30), 'white')
    panel.paste(output, (0, 0))
    draw = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        draw.text((15, output.height+15+30*i), line, font=font, fill='black')
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', type=Path, help='Corresponding Florence-2 annotated PNG')
    parser.add_argument('--boxes-json', type=Path, required=True, help='JSON produced by Florence --json')
    parser.add_argument('--line-x', type=float, help='Robot center X in original image pixels; omit to click')
    parser.add_argument('--deadband', type=float, default=0, help='Center tolerance in pixels (default: 0)')
    parser.add_argument('--mirrored', action='store_true', help='Reverse turn direction for mirrored camera images')
    parser.add_argument('-o', '--output', type=Path, help='Output PNG (default: IMAGE_direction.png)')
    parser.add_argument('--json', type=Path, help='Result JSON (default: OUTPUT with .json extension)')
    args = parser.parse_args()
    if not math.isfinite(args.deadband) or args.deadband < 0:
        parser.error('--deadband must be a finite nonnegative number')
    output = args.output or args.image.with_name(args.image.stem+'_direction.png')
    result_path = args.json or output.with_suffix('.json')
    if output.suffix.lower() != '.png':
        parser.error('Output image must be PNG')
    paths = [args.image.resolve(), args.boxes_json.resolve(), output.resolve(), result_path.resolve()]
    if len(set(paths)) != len(paths):
        parser.error('Input and output paths must all be different')
    try:
        with Image.open(args.image) as source:
            image = source.convert('RGB')
        size, regions = read_boxes(args.boxes_json, image.size)
        x = args.line_x if args.line_x is not None else pick_line(image, size)
        if not math.isfinite(x) or not 0 <= x < size[0]:
            raise ValueError(f'--line-x must be >= 0 and < {size[0]}')
        result = evaluate(regions, x, args.deadband, args.mirrored)
        result.update({'image_size': size, 'image': str(args.image),
                       'boxes_json': str(args.boxes_json), 'selection_rule': 'largest_bbox_area',
                       'comparison_rule': 'bbox_center_x', 'tie_rule': 'first_in_json'})
        annotated = draw_result(image, size, result)
        output.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False))
        print(f'Saved: {output.resolve()}\nSaved: {result_path.resolve()}', file=sys.stderr)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
