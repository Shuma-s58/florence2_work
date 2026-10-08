#!/usr/bin/env python3
"""白い紙(ラベル)を検出 → 台形補正 → モンタージュ画像1枚を最終出力する。

使い方:
    pip install opencv-python-headless numpy
    python crop_labels_montage.py delivery_box_2026.jpg -o montage.png

オプション:
    --scale 2.0       モンタージュ内の各切り出しの拡大率
    --debug DIR       検出結果の重ね描き(overlay.jpg)とマスク(mask.png)をDIRに保存
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# --- 調整用パラメータ -------------------------------------------------------
S_MAX = 60            # 彩度の上限 (小さいほど「真っ白」だけを拾う)
V_MIN = 175           # 明度の下限
AREA_MIN, AREA_MAX = 5000, 60000   # 紙の面積の範囲 (px^2)
SOLIDITY_MIN = 0.85   # 輪郭面積 / 凸包面積
FILL_MIN = 0.85       # 輪郭面積 / 近似四角形の面積
TOP_EXTEND = 0.13     # 上辺を外側へ広げる割合 (黒いロゴの取りこぼし対策)
GAP = 12              # モンタージュのタイル間の隙間 (px)
BG = (128, 128, 128)  # モンタージュの背景色 (BGR)
# ---------------------------------------------------------------------------


def order_corners(p):
    """4点を 左上→右上→右下→左下 の順に並べる。"""
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -np.argmin(p.sum(1)), axis=0).astype(np.float32)


def find_paper_quads(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, V_MIN), (180, S_MAX, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if not AREA_MIN < area < AREA_MAX:
            continue
        hull = cv2.convexHull(c)
        if area / cv2.contourArea(hull) < SOLIDITY_MIN:
            continue
        peri = cv2.arcLength(hull, True)
        approx = None
        for eps in (0.02, 0.03, 0.04, 0.05):
            t = cv2.approxPolyDP(hull, eps * peri, True)
            if len(t) == 4:
                approx = t
                break
        if approx is None:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        if area / cv2.contourArea(quad) < FILL_MIN:
            continue
        quads.append(order_corners(quad))

    # 画像の左→右の順に並べて、番号を安定させる
    quads.sort(key=lambda q: q[:, 0].mean())
    return quads, mask


def extend_top(q, ratio, w, h):
    q = q.copy()
    q[0] = q[0] + (q[0] - q[3]) * ratio
    q[1] = q[1] + (q[1] - q[2]) * ratio
    return q.clip([0, 0], [w - 1, h - 1]).astype(np.float32)


def warp(img, q):
    w = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def make_montage(crops, scale):
    """切り出し画像を拡大して番号を付け、横一列に並べる。"""
    tiles = []
    for i, im in enumerate(crops, 1):
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.putText(im, str(i), (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        tiles.append(im)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [
        cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, GAP, cv2.BORDER_CONSTANT, value=BG)
        for t in tiles
    ]
    return np.hstack(tiles)[:, :-GAP]   # 末尾の余白を削る


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="montage.png")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--debug", metavar="DIR")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"画像を読めません: {args.image}")
    H, W = img.shape[:2]

    quads, mask = find_paper_quads(img)
    if not quads:
        raise SystemExit("紙が検出できませんでした。S_MAX / V_MIN を調整してください。")

    quads = [extend_top(q, TOP_EXTEND, W, H) for q in quads]
    crops = [warp(img, q) for q in quads]
    cv2.imwrite(args.output, make_montage(crops, args.scale))
    print(f"{len(crops)} 枚検出 -> {args.output}")

    if args.debug:
        d = Path(args.debug)
        d.mkdir(parents=True, exist_ok=True)
        vis = img.copy()
        for i, q in enumerate(quads, 1):
            cv2.polylines(vis, [q.astype(int)], True, (0, 255, 0), 3)
            cv2.putText(vis, str(i), tuple(q[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(d / "overlay.jpg"), vis)
        cv2.imwrite(str(d / "mask.png"), mask)


if __name__ == "__main__":
    main()
