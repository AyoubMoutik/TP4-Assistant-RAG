"""Recompute only RAGAS answer relevancy from saved answers and patch the CSV/summary files."""
import json
import sys

import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate import get_judge
from src.rag import get_embeddings


def main(cfg: str, summary_name: str):
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ResponseRelevancy

    answers = json.loads((RESULTS_DIR / f"answers_{cfg}.json").read_text(encoding="utf-8"))
    ds = EvaluationDataset.from_list([{"user_input": a["question"], "response": a["answer"],
                                       "retrieved_contexts": a["contexts"]} for a in answers])
    res = evaluate(ds, metrics=[ResponseRelevancy(strictness=1)], llm=LangchainLLMWrapper(get_judge()),
                   embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
                   run_config=RunConfig(max_workers=2, max_retries=10, max_wait=90))
    vals = res.to_pandas()["answer_relevancy"].tolist()
    csv = RESULTS_DIR / f"ragas_{cfg}.csv"
    df = pd.read_csv(csv)
    df["answer_relevancy"] = vals
    df.to_csv(csv, index=False, encoding="utf-8")
    sp = RESULTS_DIR / summary_name
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["ragas"][cfg]["answer_relevancy"] = round(float(pd.Series(vals).mean()), 3)
    sp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(cfg, "answer_relevancy =", s["ragas"][cfg]["answer_relevancy"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
