"""Central configuration (paths, models, seeds)."""
import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "Course" / "course_merged.pdf"
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CAPTIONS_PATH = DATA_DIR / "figure_captions.json"
GOLDEN_PATH = DATA_DIR / "golden_set.json"

# Multilingual embedding model: French questions <-> English slides in one space.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
# Free-tier Gemini API: each model has its own daily quota (gemini-2.5-flash = 20 req/day only).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")  # different model than the generator
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "gemini-3.5-flash-lite")
TOP_K = int(os.getenv("TOP_K", "5"))
SEED = 42

# The merged PDF is the concatenation of these lectures (page ranges are 1-based, inclusive).
LECTURES = [
    (1, 39, "Lecture 1 — From Words to Vectors"),
    (40, 69, "Lecture 2A — The Attention Mechanism"),
    (70, 118, "Lecture 2B — Positional Encoding, Masking, Cross & Multi-Head Attention"),
    (119, 155, "Lecture 3 — The Transformer Block (Encoder/Decoder)"),
    (156, 192, "Part 1 — LLM Foundations & Inference"),
    (193, 252, "Part 2 — Prompt Engineering & Evaluation"),
]


def lecture_for_page(page: int) -> str:
    for start, end, name in LECTURES:
        if start <= page <= end:
            return name
    return "Unknown"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
