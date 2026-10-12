# florence2_work
 
Florence-2 を用いたラベル認識・箱の検出と、ロボット中心線に対する左右判定。
 
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
- `florence2_phrase.py`:`torch`、`transformers`、`pillow`
- `bbox_direction.py`:`pillow`（クリックで中心線を指定する場合は `tkinter` も必要）
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
### 3. Florence-2 による箱の検出（florence2_phrase.py）

`<CAPTION_TO_PHRASE_GROUNDING>` を使い、文章で指定した箱の位置を検出します。文字認識は行いません。箱の位置と左右方向を調べる場合は、モンタージュ画像ではなく、カメラで撮影した元画像を入力してください。

```bash
python florence2_phrase.py ../delivery_box_2026.jpg \
  -o boxes.png --json boxes.json
```

既存のローカルモデルを使う場合:

```bash
python florence2_phrase.py ../delivery_box_2026.jpg \
  --model Florence2-base/ --device cuda \
  -o boxes.png --json boxes.json
```

既定の検出対象は `A blue box.` です。文章は `--text` で変更できます。

```bash
python florence2_phrase.py ../delivery_box_2026.jpg \
  --text "A blue plastic storage box." \
  -o boxes.png --json boxes.json
```

| 引数 | 説明 |
|---|---|
| `image`（位置引数） | カメラで撮影した入力画像 |
| `-o`, `--output` | 出力 PNG（省略時は `<入力名>_blue_box.png`） |
| `--text` | 検出する対象の英文（既定 `A blue box.`） |
| `--model` | Transformers 形式のモデル ID またはローカルディレクトリ（既定 `florence-community/Florence-2-large`） |
| `--device` | `auto` / `cpu` / `cuda`（既定 `auto`） |
| `--font`, `--font-size` | 結果表示用のフォントとサイズ（サイズの既定 `20`） |
| `--max-new-tokens` | 生成トークン数の上限（既定 `2048`） |
| `--json` | 検出結果の保存先。次の左右判定で使用するため指定する |

#### 出力

- **結果画像（PNG）**:検出した各箱の bbox と番号を元画像に重ね、下部に結果を表示します。元画像部分は拡大・縮小しません。
- **標準出力**:`detected`、`count`、`regions` を JSON で出力します。
- **JSON ファイル**:元画像サイズ、使用モデル、検出対象の文章、検出有無、個数、全 bbox を保存します。

各領域は次の形式です。座標は EXIF の向きを補正した元画像のピクセル座標で、原点は左上です。

```json
{
  "id": 1,
  "label": "A blue box",
  "bbox": [100.0, 120.0, 400.0, 420.0]
}
```

`bbox` は `[xmin, ymin, xmax, ymax]`、つまり左上と右下の座標です。複数の候補はすべて保存します。`detected: true` は有効な候補が1件以上返ったことを示します。画素の色を確認する処理は含まれていないため、「青い」と指定しても別の色の箱が候補になる場合があります。

### 4. 最大 bbox とロボット中心線の左右判定（bbox_direction.py）

`florence2_phrase.py` の結果画像と、同じ推論で保存した JSON を読み込みます。Florence-2 の再推論は行いません。描画された枠の画像解析はせず、JSON の bbox 座標を使って判定します。

カメラの取り付け位置に合わせて、ロボット中心を表す縦線を指定してください。画像中央とは別の位置に設定できます。

#### 画像をクリックして中心線を設定する

```bash
python bbox_direction.py boxes.png --boxes-json boxes.json
```

画像上でロボット中心の位置をクリックし、Enter キーまたは `Confirm` ボタンで確定します。Esc キーまたはウィンドウを閉じるとキャンセルします。

GUI を使う場合は Ubuntu の Tkinter が必要です。未インストールなら次を実行してください。

```bash
sudo apt install python3-tk
```

#### 中心線の位置を数値で固定する

例えば、ロボット中心が元画像の左端から 620 px の位置なら、次のように実行します。

```bash
python bbox_direction.py boxes.png \
  --boxes-json boxes.json --line-x 620 \
  -o direction.png --json direction.json
```

クリックで選んだ座標は判定結果の `line_x` に保存されます。次回以降はその値を `--line-x` に指定してください。画像の解像度やカメラの取り付け位置を変えた場合は、中心線を再確認してください。

| 引数 | 説明 |
|---|---|
| `image`（位置引数） | Florence-2 が出力した bbox 入り画像 |
| `--boxes-json` | 同じ画像に対応する検出結果 JSON（必須） |
| `--line-x` | ロボット中心の X 座標。元画像のピクセル単位。省略時は GUI でクリック指定 |
| `--deadband` | 中心とみなす左右の許容幅（px、既定 `0`） |
| `--mirrored` | 鏡像画像の場合に、左右判定から求める旋回方向を反転する |
| `-o`, `--output` | 判定画像 PNG（省略時は `<入力名>_direction.png`） |
| `--json` | 判定結果 JSON（省略時は出力 PNG の拡張子を `.json` に変更したパス） |

#### 選択・判定の方法

1. 各 bbox の面積を `(xmax - xmin) × (ymax - ymin)` で計算します。
2. 最大面積の bbox を1つ選択します。同じ面積なら JSON 内で先に現れた候補を選びます。
3. 選択した bbox の中心 X 座標を `(xmin + xmax) / 2` で計算します。
4. `offset_x_px = center_x - line_x` により左右を判定します。

| 条件 | `side` | 通常の非鏡像画像での `turn_direction` |
|---|---|---|
| `offset_x_px < -deadband` | `left` | `left`（左を向く） |
| `offset_x_px > deadband` | `right` | `right`（右を向く） |
| 上記以外 | `center` | `center`（中心に一致、または許容幅内） |
| bbox が0件 | `not_detected` | `undetermined`（方向未確定） |

bbox が中心線をまたいでいても、左右判定には **bbox の中心**を使います。`crosses_line` には、bbox が中心線をまたぐかどうかを別途保存します。

中心線の左右 10 px を許容範囲にする例:

```bash
python bbox_direction.py boxes.png \
  --boxes-json boxes.json --line-x 620 --deadband 10
```

#### 出力

- **結果画像（PNG）**:ロボット中心線をマゼンタ、選択した最大 bbox と中心点を黄色で描画します。画像下部に判定結果を追加します。
- **JSON ファイル・標準出力**:`detected`、`candidate_count`、`line_x`、`side`、`turn_direction`、選択した bbox、面積、中心座標、中心線との差を保存・出力します。

`selected` 内の主な値:

| キー | 内容 |
|---|---|
| `id`, `label`, `bbox` | 選択した bbox の情報 |
| `area_px2` | bbox 面積（px²） |
| `center_x`, `center_y` | bbox の中心座標（px） |
| `offset_x_px` | 中心線からの横方向の差。負なら左、正なら右 |
| `crosses_line` | bbox が中心線をまたぐ場合に `true` |

bbox 未検出時は `selected` が `null` になり、旋回方向は決まりません。このスクリプトは左右方向の判定までを行い、ロボットへの回転指令や回転角度の計算は行いません。

入力画像はリサイズや切り抜きをせず、推論時に生成した JSON と組み合わせて使用してください。元スクリプトの画像下部に追加された説明欄は、bbox 座標や中心線の範囲に含めません。`--line-x` 指定時は GUI 不要です。

## 参考
 
- [Transformers: Florence-2](https://huggingface.co/docs/transformers/model_doc/florence2)
- [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large)

