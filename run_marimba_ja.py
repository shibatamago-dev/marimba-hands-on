#!/usr/bin/env python3
"""Run Marimba only on Japanese clips in audio/ja_*.wav, merge into results_ja.json"""
import json
from pathlib import Path
from marimba import Marimba, Question

BASE = Path(__file__).parent
clips = sorted((BASE / "audio").glob("ja_*.wav"))
print("clips:", [c.name for c in clips])
model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
questions = [
    Question.score("Has the speaker finished?"),
    Question.noul("Is someone speaking?"),
    Question.choice("What language?", ["en", "ja", "pt", "es", "zh", "Other"]),
    Question.noul('Did someone say "marimba"?'),
    Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
]
out = {}
for c in clips:
    print(f"\n=== {c.name} ===")
    rows = []
    for d in model(str(c), questions):
        r = d.to_dict()
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    out[c.name] = rows
(BASE / "results_ja.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved results_ja.json")
