"""Partie B — RAG with citations (+ bonus: hybrid search, reranking, query rewriting,
conversational memory, OpenTelemetry tracing).

Usage:
    python -m src.rag "Pourquoi divise-t-on les scores d'attention par racine de d_k ?"
    python -m src.rag --mode hybrid --translate --rerank "..."
    python -m src.rag --chat           # conversational mode (memory + query rewriting)
"""
import argparse
import hashlib
import json
import os
import re
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from src.config import CHUNKS_PATH, DATA_DIR, EMBED_MODEL, INDEX_DIR, LLM_MODEL, TOP_K

REFUSAL = "Je ne trouve pas cette information dans le cours."

# --------------------------------------------------------------------------- tracing
class TimingExporter(SpanExporter):
    """Keeps finished spans in memory so we can report per-step latency."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def durations_ms(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for s in self.spans:
            out.setdefault(s.name, []).append((s.end_time - s.start_time) / 1e6)
        return out


TIMINGS = TimingExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(TIMINGS))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("rag-assistant")


# --------------------------------------------------------------------------- models
class E5Embeddings(Embeddings):
    """multilingual-e5 needs 'query: ' / 'passage: ' prefixes."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = model_name  # RAGAS reads .model as a string
        self._st = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return self._st.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                                 batch_size=16, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self._st.encode(f"query: {text}", normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> E5Embeddings:
    return E5Embeddings()


def thinking_kwargs(model: str) -> dict:
    """Minimise 'thinking' (latency/quota): parameter name differs per model family."""
    if model.startswith("gemini-2.5"):
        return {"thinking_budget": 0}
    if model.startswith("gemini-3") or "latest" in model:
        return {"thinking_level": "low"}
    return {}  # gemma


@lru_cache(maxsize=8)
def get_llm(model: str = LLM_MODEL, temperature: float = 0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=2, timeout=120,
                                  **thinking_kwargs(model))


def to_text(content) -> str:
    if isinstance(content, list):  # Gemini 3 returns content blocks
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content.strip()


# Fallback chain used ONLY when the current model's free daily quota is exhausted.
FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "LLM_FALLBACKS", "gemini-3.1-flash-lite,gemini-flash-lite-latest,gemma-4-31b-it").split(",") if m.strip()]
_exhausted: set[str] = set()
LAST_MODEL_USED = {"name": LLM_MODEL}


_CACHE_PATH = DATA_DIR / "llm_cache.json"
_cache: dict | None = None


def _cache_get_set(key: str, value: str | None = None):
    global _cache
    if _cache is None:
        _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8")) if _CACHE_PATH.exists() else {}
    if value is None:
        return _cache.get(key)
    _cache[key] = value
    _CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=0), encoding="utf-8")


def llm_call(prompt, model: str = LLM_MODEL, use_cache: bool = True) -> str:
    """LLM call (temperature 0) with a disk cache and back-off for free-tier 429 errors."""
    key = hashlib.sha256(json.dumps([model, prompt], ensure_ascii=False).encode()).hexdigest()
    if use_cache and (hit := _cache_get_set(key)) is not None:
        LAST_MODEL_USED["name"] = model
        return hit
    for candidate in [model] + [m for m in FALLBACK_MODELS if m != model]:
        if candidate in _exhausted:
            continue
        for attempt in range(6):
            try:
                content = to_text(get_llm(candidate).invoke(prompt).content)
                if use_cache:
                    _cache_get_set(key, content)  # cached under the requested model's key
                LAST_MODEL_USED["name"] = candidate
                return content
            except Exception as e:
                msg = f"{type(e).__name__} {e}"
                if "per day" in msg.lower() or "PerDay" in msg:
                    print(f"  [quota] daily quota exhausted for {candidate} -> next model")
                    _exhausted.add(candidate)
                    break
                if any(s in msg.lower() for s in ("429", "resource_exhausted", "503", "unavailable",
                                                  "timeout", "connect", "10054")):
                    wait = 15 * (attempt + 1)
                    print(f"  [retry] {candidate}: {type(e).__name__}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise
    raise RuntimeError("All LLMs unavailable (quota or errors)")


@lru_cache(maxsize=1)
def get_store():
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_bm25():
    from rank_bm25 import BM25Okapi

    docs = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.pop("text")
            docs.append(Document(page_content=text, metadata=r))
    bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
    return bm25, docs


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-base", device="cpu")  # multilingual cross-encoder


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# --------------------------------------------------------------------------- prompts
SYSTEM = """Tu es l'assistant pédagogique du module « Transformers & LLMs » (ENSAM Casablanca).
Les extraits du cours ci-dessous sont en anglais ; tu réponds TOUJOURS en français.

Règles :
1. Réponds UNIQUEMENT à partir des extraits fournis. N'ajoute aucune connaissance externe.
2. Après chaque phrase factuelle, cite l'extrait utilisé avec son numéro entre crochets, ex. [2] ou [1][3].
3. Si les extraits ne permettent pas de répondre, réponds exactement : « {refusal} »
4. Sois concis (3 à 6 phrases), précis, et garde les termes techniques (Query, Key, softmax...)."""

REWRITE = """You turn a student's question about a Transformers & LLMs course into ONE short
English search query for the (English) course slides. Resolve pronouns using the chat history.
Keep technical terms. Output only the query.

Chat history:
{history}

Question: {question}
English search query:"""


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        blocks.append(f"[{i}] ({m['lecture']}, slide {m['page']} — {m['title']})\n{d.page_content}")
    return "\n\n".join(blocks)


def citation_label(d: Document) -> str:
    m = d.metadata
    return f"{m['lecture']}, p. {m['page']} — « {m['title']} »"


# --------------------------------------------------------------------------- pipeline
class RAGAssistant:
    def __init__(self, mode: str = "dense", translate: bool = False, rerank: bool = False,
                 k: int = TOP_K, fetch_k: int = 20):
        assert mode in ("dense", "hybrid")
        self.mode, self.translate, self.rerank, self.k, self.fetch_k = mode, translate, rerank, k, fetch_k
        self.history: list[tuple[str, str]] = []
        # warm-up so that model loading is not counted in the traced latencies
        get_store()
        if mode == "hybrid":
            get_bm25()
        if rerank:
            get_reranker()

    @property
    def name(self) -> str:
        return "+".join([self.mode] + (["translate"] if self.translate else [])
                        + (["rerank"] if self.rerank else []))

    # -- query rewriting (FR -> EN search query, resolves conversational references)
    def rewrite(self, question: str) -> str:
        with tracer.start_as_current_span("rag.query_rewrite"):
            hist = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in self.history[-3:]) or "(none)"
            return llm_call(REWRITE.format(history=hist, question=question))

    # -- retrieval
    def retrieve(self, query: str) -> list[Document]:
        n = self.fetch_k if (self.rerank or self.mode == "hybrid") else self.k
        with tracer.start_as_current_span("rag.retrieve") as span:
            dense = get_store().similarity_search(query, k=n)
            if self.mode == "hybrid":
                bm25, all_docs = get_bm25()
                scores = bm25.get_scores(tokenize(query))
                top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
                sparse = [all_docs[i] for i in top]
                # Reciprocal Rank Fusion
                fused: dict[str, list] = {}
                for ranking in (dense, sparse):
                    for rank, d in enumerate(ranking):
                        cid = d.metadata["chunk_id"]
                        fused.setdefault(cid, [0.0, d])[0] += 1.0 / (60 + rank)
                docs = [d for _, d in sorted(fused.values(), key=lambda x: -x[0])][:n]
            else:
                docs = dense
            span.set_attribute("rag.n_candidates", len(docs))
        if self.rerank:
            with tracer.start_as_current_span("rag.rerank"):
                scores = get_reranker().predict([(query, d.page_content) for d in docs])
                docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
        return docs[: self.k]

    # -- generation
    def answer(self, question: str, conversational: bool = False) -> dict:
        with tracer.start_as_current_span("rag.request") as root:
            root.set_attribute("rag.config", self.name)
            need_rewrite = self.translate or (conversational and self.history)
            query = self.rewrite(question) if need_rewrite else question
            docs = self.retrieve(query)
            with tracer.start_as_current_span("rag.generate"):
                messages = [
                    ("system", SYSTEM.format(refusal=REFUSAL)),
                    ("human", f"Extraits du cours :\n\n{format_context(docs)}\n\nQuestion : {question}"),
                ]
                text = llm_call(messages)
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(docs)})
        sources = [f"[{i}] {citation_label(docs[i - 1])}" for i in used]
        if conversational:
            self.history.append((question, text))
        return {
            "question": question,
            "search_query": query,
            "answer": text,
            "sources": sources,
            "contexts": [d.page_content for d in docs],
            "pages": [d.metadata["page"] for d in docs],
            "refused": REFUSAL.rstrip(".") in text,
            "model": LAST_MODEL_USED["name"],
        }


def pretty(r: dict) -> str:
    out = f"\nQ : {r['question']}\n"
    if r["search_query"] != r["question"]:
        out += f"(requête de recherche : {r['search_query']})\n"
    out += f"\n{r['answer']}\n\nSources :\n" + ("\n".join(r["sources"]) or "  (aucune)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", default="dense", choices=["dense", "hybrid"])
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--chat", action="store_true", help="conversational mode")
    args = ap.parse_args()
    rag = RAGAssistant(args.mode, args.translate, args.rerank, args.k)

    if args.chat:
        print("Assistant du cours — tapez 'q' pour quitter.")
        while (q := input("\nVous > ").strip()) not in ("q", "quit", ""):
            print(pretty(rag.answer(q, conversational=True)))
    else:
        print(pretty(rag.answer(args.question or "Qu'est-ce que l'attention multi-têtes ?")))
    print("\nTrace (ms):", {k: round(sum(v), 1) for k, v in TIMINGS.durations_ms().items()})


if __name__ == "__main__":
    main()
