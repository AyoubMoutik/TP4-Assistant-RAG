"""Parties C & D — Evaluation.

1. Retrieval metrics (no judge LLM): Hit@k / MRR against the golden set's reference slides,
   for every retrieval configuration (dense, hybrid, +translate, +rerank).
2. RAGAS (Gemini judge): faithfulness, answer relevancy, context precision, context recall.
3. "Insufficient context" test: out-of-scope questions -> does the system refuse?
4. Per-step latency from the OpenTelemetry spans.

Usage:
    python -m src.evaluate --retrieval-only
    python -m src.evaluate --configs dense dense+translate
"""
import argparse
import json
import os
import statistics
import time

import pandas as pd

from src.config import GOLDEN_PATH, JUDGE_MODEL, RESULTS_DIR, TOP_K, set_seed
from src.rag import TIMINGS, RAGAssistant, get_embeddings

ALL_CONFIGS = ["dense", "hybrid", "dense+translate", "hybrid+translate"]  # "+rerank" also supported


def make_assistant(cfg: str) -> RAGAssistant:
    parts = cfg.split("+")
    return RAGAssistant(mode=parts[0], translate="translate" in parts, rerank="rerank" in parts, k=TOP_K)


def load_golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 1. retrieval
def retrieval_metrics(cfg: str, golden: list[dict]) -> dict:
    rag = make_assistant(cfg)
    hits, rr, rows = [], [], []
    for g in golden:
        q = rag.rewrite(g["question"]) if rag.translate else g["question"]
        pages = [d.metadata["page"] for d in rag.retrieve(q)]
        ref = set(g["reference_pages"])
        rank = next((i + 1 for i, p in enumerate(pages) if p in ref), None)
        hits.append(1 if rank else 0)
        rr.append(1 / rank if rank else 0)
        rows.append({"id": g["id"], "query": q, "retrieved_pages": pages, "reference": sorted(ref), "rank": rank})
    return {"config": cfg, f"hit@{TOP_K}": statistics.mean(hits), "mrr": statistics.mean(rr), "details": rows}


# --------------------------------------------------------------------------- 2. RAGAS
def generate_answers(cfg: str, golden: list[dict]) -> list[dict]:
    rag = make_assistant(cfg)
    out = []
    for g in golden:
        r = rag.answer(g["question"])
        r["reference"] = g["reference"]
        r["id"] = g["id"]
        out.append(r)
        print(f"  [{cfg}] Q{g['id']} -> pages {r['pages']}")
    (RESULTS_DIR / f"answers_{cfg}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def get_judge():
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.rag import thinking_kwargs

    return ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0, max_retries=10, timeout=180,
                                  **thinking_kwargs(JUDGE_MODEL))


def run_ragas(cfg: str, answers: list[dict]) -> pd.DataFrame:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    ds = EvaluationDataset.from_list([
        {"user_input": a["question"], "response": a["answer"],
         "retrieved_contexts": a["contexts"], "reference": a["reference"]}
        for a in answers
    ])
    judge = LangchainLLMWrapper(get_judge(), cache=DiskCacheBackend(cache_dir=str(RESULTS_DIR / ".ragas_cache")))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(),
               LLMContextRecall()]
    # max_workers=2 keeps us under the free-tier requests-per-minute limit
    res = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb,
                   run_config=RunConfig(max_workers=2, max_retries=10, max_wait=90, timeout=300))
    df = res.to_pandas()
    df.insert(0, "id", [a["id"] for a in answers])
    df.to_csv(RESULTS_DIR / f"ragas_{cfg}.csv", index=False, encoding="utf-8")
    return df


# --------------------------------------------------------------------------- 3. refusals
def refusal_test(cfg: str, oos: list[dict]) -> list[dict]:
    rag = make_assistant(cfg)
    rows = []
    for o in oos:
        r = rag.answer(o["question"])
        rows.append({"question": o["question"], "answer": r["answer"], "refused": r["refused"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["dense", "dense+translate"])
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()
    set_seed()
    RESULTS_DIR.mkdir(exist_ok=True)
    golden = load_golden()
    summary_path = RESULTS_DIR / ("summary.json" if not args.skip_retrieval else f"summary_{'_'.join(args.configs)}.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    # 1. retrieval metrics for every configuration
    print("== Retrieval metrics")
    summary["retrieval"] = {}
    for cfg in ([] if args.skip_retrieval else ALL_CONFIGS):
        m = retrieval_metrics(cfg, golden["answerable"])
        summary["retrieval"][cfg] = {k: v for k, v in m.items() if k != "details"}
        (RESULTS_DIR / f"retrieval_{cfg}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {cfg:28s} hit@{TOP_K}={m[f'hit@{TOP_K}']:.2f}  mrr={m['mrr']:.3f}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.retrieval_only:
        return

    # 2. RAGAS
    summary.setdefault("ragas", {})
    for cfg in args.configs:
        print(f"== RAGAS for {cfg}")
        TIMINGS.spans.clear()
        answers = generate_answers(cfg, golden["answerable"])
        lat = {k: round(statistics.mean(v), 1) for k, v in TIMINGS.durations_ms().items()}
        df = run_ragas(cfg, answers)
        cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy",
                                                "llm_context_precision_with_reference", "context_recall")]
        summary["ragas"][cfg] = {c: round(float(df[c].mean()), 3) for c in cols}
        summary["ragas"][cfg]["latency_ms_mean_per_step"] = lat
        print(json.dumps(summary["ragas"][cfg], indent=2))
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. refusals on out-of-scope questions
    print("== Out-of-scope (insufficient context) test")
    rows = refusal_test(args.configs[-1], golden["out_of_scope"])
    summary["out_of_scope"] = rows
    for r in rows:
        print(f"  refused={r['refused']}  {r['question']}\n    -> {r['answer'][:200]}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
