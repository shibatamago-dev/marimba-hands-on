# Marimba ハンズオン

Charcot AI「Marimba」をCPUで試した再現用リポジトリ。

Marimbaは短い音声ウィンドウ（最後の8秒）に対して、文字起こしではなく「型付き判定」を返すモデル。
凍結した Qwen3-ASR-0.6B バックボーン + 小さな readout という構成。

- モデル: https://huggingface.co/charcot-ai/marimba
- 論文: arXiv:2611.18330（公表値は自社計測）
- ライセンス: モデル・コードは Apache-2.0 / このリポジトリのコードは MIT（LICENSE参照）

## 結果の要点（詳細は RESULTS.md）

| クリップ | 発話終了 | 言語 |
|---|---|---|
| 公式サンプル（英語） | 0.899 | en 0.9998 |
| 無音3秒 | 0.980（発話有無は false / 0.00002） | es 0.872 ※閉集合の強制選択 |
| 日本語・言い切り | **0.986** | ja 0.9997 |
| 日本語・言いかけ | **0.134** | ja 0.9996 |
| 日本語「マリンバ…」 | 0.994 | ja 0.9995（英語単語 "marimba" 検出は false） |

注意点：言語判定は閉集合の強制選択（無音でも何か選ぶ）、話者系は有効8秒未満で null、単語検出は未較正の仮説割合で表記依存。
検証は全て合成音声（公式サンプル＋Kokoro TTS）。肉声・騒音環境は未検証。
実行時間は環境依存のため、モデルの速度としては扱っていません。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install "marimba-audio[torch]"
# 日本語TTS生成用（任意）
.venv/bin/pip install kokoro "misaki[ja]"
.venv/bin/python -m unidic download
```

※ pip の一時領域が小さい環境では `TMPDIR` を容量のある場所に向けてください（torch が約888MBあります）。

## 実行

```bash
# 無音クリップ生成（ffmpegが必要）
mkdir -p audio
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 3 audio/silence_3s.wav

# 日本語TTS（Kokoro、ローカル生成）
.venv/bin/python make_tts_kokoro.py

# 公式サンプル＋ローカル音声
.venv/bin/python run_marimba.py        # -> results.json

# 日本語のみ追加推論
.venv/bin/python run_marimba_ja.py     # -> results_ja.json
```

## 判定の種類

| 質問 | 種類 | 返る値 |
|---|---|---|
| Has the speaker finished? | score | 終了確率 0-1 |
| Is someone speaking? | noul (yes/no) | bool + probability |
| What language? [en, ja, pt, es, zh, Other] | choice | 選ばれた言語 + 各確率 |
| Did someone say "marimba"? | noul | bool（termスコアは未較正） |
| How many speakers? [0,1,2,3+] | choice | 話者数（8秒未満は null） |
