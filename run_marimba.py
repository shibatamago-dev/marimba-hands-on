#!/usr/bin/env python3
"""
Marimba hands-on: typed decisions on short audio windows.
- Official HF Space examples
- Silence
- Japanese / English TTS clips (if present in audio/)
Outputs JSON + prints table. Timing is environment-dependent, not reported as model speed.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from marimba import Marimba, Question

BASE = Path(__file__).parent
AUDIO_DIR = BASE / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS = BASE / "results.json"

def download_example(name: str) -> str:
    p = hf_hub_download("charcot-ai/marimba", f"space/examples/{name}", repo_type="space")
    print(f"downloaded {name} -> {p}")
    return p

def main():
    clips = {}

    # 1) Official examples mentioned in docs
    for name in ["01_turn_end.wav", "03_term.wav"]:
        try:
            clips[name] = download_example(name)
        except Exception as e:
            print(f"WARN failed {name}: {e}")

    # Also try 02 if exists
    try:
        clips["02_example.wav"] = download_example("02_voice_activity.wav")
    except Exception as e:
        print(f"note 02 not fetched: {e}")

    # 2) Local clips if present
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        clips[wav.name] = str(wav)

    if not clips:
        print("No clips found")
        sys.exit(1)

    print(f"\nClips to test: {list(clips.keys())}")
    print("Loading model (CPU, float32) - first load downloads Qwen3-ASR-0.6B backbone...")
    model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
    print("Model loaded.\n")

    # Language options include Japanese - key check for this hands-on
    questions = [
        Question.score("Has the speaker finished?"),
        Question.noul("Is someone speaking?"),
        Question.choice("What language?", ["en", "ja", "pt", "es", "zh", "Other"]),
        Question.noul('Did someone say "marimba"?'),
        Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
    ]

    all_results = {}
    for name, path in clips.items():
        print(f"\n=== {name} : {path} ===")
        try:
            decisions = model(path, questions)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            all_results[name] = {"error": str(e)}
            continue
        rows = []
        for d in decisions:
            row = d.to_dict()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        all_results[name] = rows

    RESULTS.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS}")

if __name__ == "__main__":
    main()
