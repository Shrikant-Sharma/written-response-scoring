"""Shared names: where things live, the score scale of each prompt, and small helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW, WORK, RESULTS, SQL = ROOT / "data" / "raw", ROOT / "data" / "work", ROOT / "results", ROOT / "sql"
SEED = 20261122
DATA_FILE = "training_set_rel3.tsv"
# The lowest and highest final score a response can get, per prompt, from the publisher's set descriptions.
SCALE = {1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)}
# One rater's own scale per prompt (the final score of prompts 1, 7 and 8 adds raters up).
RATER_SCALE = {1: (1, 6), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 15), 8: (0, 30)}
PROMPTS = sorted(SCALE)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
