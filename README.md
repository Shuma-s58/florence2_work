# florence2_work
 
Florence-2 を用いたラベル認識。
 
写真に写っている白いラベル(紙)を検出して正面向きに切り出し、1枚のモンタージュ画像にまとめたうえで、Florence-2 の OCR(`<OCR_WITH_REGION>`)で文字と位置を読み取ります。
 
```
写真 ──[crop_labels_montage.py]──▶ モンタージュ画像 ──[florence2_ocr.py]──▶ OCR結果画像(+ JSON)
```
 
## 必要環境
 
動作確認した環境は次のとおりです。
 
| 項目 | バージョン |
|---|---|
| Python | 3.10(仮想環境 `.venv`) |
| torch | 2.6.0(CUDA 12.6 版 `+cu126`) |
| transformers | 4.57.1 |
| pillow | 10.4.0 |
| opencv-python | 4.11.0.86 |
| numpy | 1.26.4 |
 
- `florence2_ocr.py`:`torch`、`transformers`、`pillow`
- `crop_labels_montage.py`:`opencv-python`、`numpy`
- `--device cuda` で実行するには、NVIDIA GPU と CUDA 対応の PyTorch が必要です。
### インストール
 
```bash
python3 -m venv .venv
source .venv/bin/activate
 
# PyTorch(CUDA 12.6 版)
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
 
# そのほかのライブラリ
python -m pip install -r requirements.txt
```
 
CPU のみで実行する場合は、PyTorch を `python -m pip install torch==2.6.0` でインストールし、実行時に `--device cpu` を指定してください。
 
## 使い方
 
### 1. ラベルの切り出しとモンタージュ作成
 
```bash
python crop_labels_montage.py ../delivery_box_2026.jpg -o montage_check.png
```
 
| 引数 | 説明 |
|---|---|
| `image`(位置引数) | 入力画像 |
| `-o`, `--output` | 出力するモンタージュ画像(省略時は `montage.png`) |
| `--scale` | モンタージュ内の各切り出しの拡大率(既定 `2.0`) |
| `--debug DIR` | 検出結果の重ね描き `overlay.jpg` と白判定マスク `mask.png` を `DIR` に保存 |
 
#### モンタージュ処理の概要
 
1. **白い紙の抽出**:HSV 色空間で「彩度が低く明るい」領域を白い紙とみなし、マスクにする。
2. **候補の絞り込み**:マスクの輪郭を四角形に近似し、面積・凸包との比・四角形らしさの条件で、壁や床などを除外する。
3. **台形補正**:検出した各四角形を透視変換し、正面から見た長方形に切り出す。
4. **モンタージュ化**:切り出した画像を拡大し、左から右の順に番号を付けて、横一列に並べて1枚の画像にする。
紙が1枚も検出できなかった場合は、エラーを出して終了します。紙の色が白に近い背景(白い壁など)では分離できないことがあります。その場合は `--debug` でマスクを確認し、スクリプト冒頭の `S_MAX`、`V_MIN`、`AREA_MIN`、`AREA_MAX` などを調整してください。
 
### 2. Florence-2 による OCR
 
```bash
python florence2_ocr.py montage.png --model Florence2-base/ --device cuda -o output_montage.png
```
 
1 の出力ファイル(例:`montage_check.png`)を入力に使う場合は、ファイル名を読み替えてください。
 
| 引数 | 説明 |
|---|---|
| `image`(位置引数) | 入力画像 |
| `-o`, `--output` | 出力する PNG のパス(省略時は `<入力名>_ocr.png`) |
| `--model` | Transformers 形式の Florence-2 のモデル ID またはローカルディレクトリ(既定 `florence-community/Florence-2-large`) |
| `--device` | `auto` / `cpu` / `cuda`(既定 `auto`) |
| `--font`, `--font-size` | 結果表示に使うフォント(TTF/OTF/TTC)とサイズ。日本語を表示するときは対応フォントを指定 |
| `--max-new-tokens` | 生成するトークン数の上限(既定 `2048`) |
| `--json` | OCR 結果を JSON でも保存するパス |
 
初回実行時、`--model` に Hugging Face のモデル ID を指定すると重みをダウンロードし、以降はキャッシュを再利用します。`Florence2-base/` のようにローカルディレクトリを指定すれば、ダウンロードは不要です。
 
#### 出力
 
- **結果画像(PNG)**:元画像に、認識した文字領域の四角形と番号を重ね、その下に認識テキストと四隅の座標を並べた画像。
- **標準出力**:領域ごとの `id`、`text`、`quad_box` を1行ずつ JSON で出力。
- **JSON ファイル**(`--json` 指定時):画像サイズ、モデル名、全領域をまとめたファイル。
#### 仕様上の注意
 
- 番号(ID)はモデルの出力順で、読み順や追跡 ID は保証されません。
- 座標は EXIF の向きを補正した入力画像のピクセル座標です(原点は左上)。
- 認識した各領域は個別に保持され、近い領域の結合は行いません。
- 大文字・小文字は認識結果のまま保持し、フィルタや大文字化はしません。
- 出力が `--max-new-tokens` に達した場合は、途中で切れた可能性があるという警告を出します。その場合は上限を増やすか、入力を小さな画像に分けてください。
## 参考
 
- [Transformers: Florence-2](https://huggingface.co/docs/transformers/model_doc/florence2)
- [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large)
