"""Partie A — Ingestion offline.

PDF -> structure-preserving chunks (one slide = one unit, with lecture / section / title
metadata) -> optional figure captions (Gemini vision) -> FAISS index.

Usage:
    python -m src.ingest              # text only
    python -m src.ingest --captions   # also describe figures with Gemini (cached)
"""
import argparse
import json
import re
import time

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (CAPTION_MODEL, CAPTIONS_PATH, CHUNKS_PATH, DATA_DIR, INDEX_DIR,
                        PDF_PATH, lecture_for_page, set_seed)

MAX_CHARS = 1500  # slides longer than this are split (e5 max = 512 tokens)
SECTION_RE = re.compile(r"^(SECTION\s+\d+.*|Section\s+\d+.*|BLOC\s+[A-Z].*|PAGES?\s+[\d–-]+\s*·.*|BRIDGE\s*·.*)$")


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def slide_title(lines: list[str]) -> tuple[str | None, str]:
    """Return (section_header, title). Titles are often wrapped on 2 lines."""
    section = None
    if lines and SECTION_RE.match(lines[0]):
        section, lines = lines[0], lines[1:]
    if not lines:
        return section, ""
    # slide titles are often wrapped: join the 2nd line only if both lines are short
    title = " ".join(lines[:2]) if len(lines) > 1 and len(lines[0]) < 50 and len(lines[1]) < 45 else lines[0]
    return section, re.sub(r"\s+", " ", title).strip()[:120]


def caption_figures(doc: fitz.Document, pages: list[int]) -> dict[str, str]:
    """Describe the figures of the given pages with Gemini vision (cached on disk)."""
    from google import genai
    from google.genai import types

    cache = json.loads(CAPTIONS_PATH.read_text(encoding="utf-8")) if CAPTIONS_PATH.exists() else {}
    client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))
    prompt = ("This is a slide from a university course on Transformers & LLMs. "
              "Describe ONLY its figures/diagrams (not the text already written) in 2-4 factual "
              "English sentences: what is drawn, the components, arrows and what they show.")
    for p in pages:
        if str(p) in cache:
            continue
        png = doc[p - 1].get_pixmap(dpi=90).tobytes("png")
        for attempt in range(5):
            try:
                r = client.models.generate_content(
                    model=CAPTION_MODEL,
                    contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
                    config=types.GenerateContentConfig(
                        temperature=0, thinking_config=(
                            types.ThinkingConfig(thinking_budget=0) if CAPTION_MODEL.startswith("gemini-2.5")
                            else types.ThinkingConfig(thinking_level="low"))),
                )
                cache[str(p)] = (r.text or "").strip()
                break
            except Exception as e:  # rate limit -> back off
                transient = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "10054", "Connect", "Timeout")
                if not any(s in f"{type(e).__name__} {e}" for s in transient):
                    raise
                print(f"  page {p}: {type(e).__name__}, retry in {15 * (attempt + 1)}s")
                time.sleep(15 * (attempt + 1))
        CAPTIONS_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  captioned page {p}")
        time.sleep(6)  # stay under the free-tier RPM
    return cache


def build_chunks(use_captions: bool) -> list[Document]:
    doc = fitz.open(PDF_PATH)
    image_pages = [i + 1 for i in range(len(doc)) if doc[i].get_images()]
    captions = {}
    if use_captions:
        print(f"Captioning {len(image_pages)} slides with figures...")
        captions = caption_figures(doc, image_pages)
    elif CAPTIONS_PATH.exists():
        captions = json.loads(CAPTIONS_PATH.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks, current_section = [], None
    for i in range(len(doc)):
        page = i + 1
        text = clean(doc[i].get_text())
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        section, title = slide_title(lines)
        if section:
            current_section = section
        lecture = lecture_for_page(page)
        fig = captions.get(str(page), "")
        body = text + (f"\n[Figure] {fig}" if fig else "")
        parts = splitter.split_text(body) if len(body) > MAX_CHARS else [body]
        for j, part in enumerate(parts):
            header = f"{lecture} | {current_section or ''} | {title}"
            chunks.append(Document(
                page_content=f"{header}\n{part}",
                metadata={
                    "chunk_id": f"p{page:03d}-{j}",
                    "source": PDF_PATH.name,
                    "page": page,
                    "lecture": lecture,
                    "section": current_section or "",
                    "title": title,
                    "has_figure": page in image_pages,
                    "language": "en",
                },
            ))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", action="store_true", help="describe figures with Gemini vision")
    args = ap.parse_args()
    set_seed()
    DATA_DIR.mkdir(exist_ok=True)

    chunks = build_chunks(args.captions)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for d in chunks:
            f.write(json.dumps({"text": d.page_content, **d.metadata}, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks written to {CHUNKS_PATH}")

    from langchain_community.vectorstores import FAISS
    from src.rag import get_embeddings

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(INDEX_DIR))
    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
