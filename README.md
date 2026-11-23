# TP4 — Assistant documentaire RAG du cours « Transformers & LLMs »

Cas d'usage 1 (assistant du cours). Corpus : `Course/course_merged.pdf` (252 slides, 6 cours, en anglais).
**Langue d'interaction : français** — questions, réponses, golden set et prompts en français ;
la recherche est cross-lingue (question FR → slides EN) grâce à un embedding multilingue.

| Composant | Choix |
|---|---|
| LLM (génération) | Gemini `gemini-3.5-flash-lite` (API gratuite), température 0 |
| Juge RAGAS | Gemini `gemini-3.1-flash-lite` (modèle différent du générateur) |
| Embedding | `intfloat/multilingual-e5-base` (local, multilingue, 100+ langues dont FR/AR) |
| Base vectorielle | FAISS (cosinus, vecteurs normalisés) |
| Chunking | 1 slide = 1 chunk (découpée si > 1500 caractères) + en-tête « cours / section / titre » + description des figures (Gemini vision) |
| top-k | 5 |
| Bonus | hybride dense+BM25 (RRF), reranking cross-encoder `BAAI/bge-reranker-base`, réécriture/traduction de requête, RAG conversationnel, traçage OpenTelemetry |

## Installation (reproductible)

```bash
uv venv .venv --python 3.11          # ou: python3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env               # puis mettre votre GOOGLE_API_KEY
```

Seed fixée à 42 (`src/config.py`), température 0, et toutes les réponses LLM sont mises en cache
(`data/llm_cache.json`, `results/.ragas_cache`) : relancer les scripts redonne les mêmes résultats.

## Exécution

```bash
# Partie A — ingestion : PDF -> chunks + métadonnées -> index FAISS
python -m src.ingest --captions        # --captions : décrit les figures avec Gemini vision (cache: data/figure_captions.json)

# Partie B — poser une question
python -m src.rag "Pourquoi divise-t-on les scores d'attention par racine de d_k ?"
python -m src.rag --mode hybrid --translate --rerank "Qu'est-ce que RoPE ?"
python -m src.rag --chat               # mode conversationnel (mémoire + réécriture de requête)

# Parties C & D — évaluation (retrieval Hit@5/MRR, RAGAS, refus hors-sujet, latences)
python -m src.evaluate --configs dense
python -m src.evaluate --configs hybrid+translate --skip-retrieval
python -m src.relevancy dense summary.json   # (si answer_relevancy = NaN)
python -m src.fill_report
```

## Structure

```
Course/course_merged.pdf   corpus
data/golden_set.json       15 Q/R de référence (FR) + pages attendues, 3 questions hors-corpus
data/chunks.jsonl          chunks + métadonnées (source, page, cours, section, titre, figure, langue)
src/ingest.py              Partie A
src/rag.py                 Partie B + bonus
src/evaluate.py            Parties C & D
results/summary.json       tous les scores ; results/ragas_*.csv : scores par question
results/answers_*.json     réponses générées avec contextes et citations
COMPTE_RENDU.md            compte-rendu rempli
```

Note : l'API Gemini gratuite est limitée en requêtes/minute ; les scripts attendent et réessaient
automatiquement en cas d'erreur 429.
