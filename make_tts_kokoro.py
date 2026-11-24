#!/usr/bin/env python3
"""Generate JA clips locally with Kokoro TTS (no external TTS server)."""
from pathlib import Path
import soundfile as sf
from kokoro import KPipeline

BASE = Path(__file__).parent / "audio"
BASE.mkdir(exist_ok=True)

CLIPS = [
    ("ja_finished.wav", "今日の会議はこれで終わります。ありがとうございました。"),
    ("ja_unfinished.wav", "えっと、それでですね、次に説明したいのは"),
    ("ja_marimba.wav", "マリンバという楽器を知っていますか。マリンバです。"),
]

def main():
    pipeline = KPipeline(lang_code="j")
    for fname, text in CLIPS:
        out = BASE / fname
        if out.exists():
            print(f"skip {fname}")
            continue
        print(f"generating {fname}: {text}")
        audios = []
        for _, _, audio in pipeline(text, voice="jf_alpha"):
            audios.append(audio)
        import numpy as np
        full = np.concatenate(audios) if len(audios) > 1 else audios[0]
        sf.write(str(out), full, 24000)
        print(f"  saved {out} {len(full)/24000:.2f}s")

if __name__ == "__main__":
    main()
