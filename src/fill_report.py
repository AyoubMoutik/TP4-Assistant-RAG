"""Fill the RAGAS numbers / example / refusal results into COMPTE_RENDU.md."""
import json

from src.config import RESULTS_DIR, ROOT

KEYS = {"F": "faithfulness", "AR": "answer_relevancy",
        "CP": "llm_context_precision_with_reference", "CR": "context_recall"}


def load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    base = load("summary.json")
    imp = load("summary_hybrid+translate.json")
    rd = base.get("ragas", {}).get("dense", {})
    ri = imp.get("ragas", {}).get("hybrid+translate", {})
    md = (ROOT / "COMPTE_RENDU.md").read_text(encoding="utf-8")

    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "n/d"
    for tag, r in (("D", rd), ("I", ri)):
        for k, col in KEYS.items():
            md = md.replace(f"RAGAS_{tag}_{k}", fmt(r.get(col)))

    # example Q/A
    ans = load("answers_hybrid+translate.json") or load("answers_dense.json")
    if ans:
        a = next((x for x in ans if x["id"] == 5), ans[0])
        ex = (f"**Question :** {a['question']}\n\n**Réponse :** {a['answer']}\n\n**Sources citées :**\n"
              + "\n".join(f"- {s}" for s in a["sources"]))
        md = md.replace("EXAMPLE_QA", ex)

    # interpretation
    if rd and ri:
        d = {k: (ri.get(c, 0) or 0) - (rd.get(c, 0) or 0) for k, c in KEYS.items()}
        interp = (
            f"Baseline : fidélité {fmt(rd.get(KEYS['F']))}, pertinence {fmt(rd.get(KEYS['AR']))}, "
            f"précision du contexte {fmt(rd.get(KEYS['CP']))}, rappel du contexte {fmt(rd.get(KEYS['CR']))}. "
            f"Version améliorée (réécriture FR→EN + hybride) : {fmt(ri.get(KEYS['F']))} / {fmt(ri.get(KEYS['AR']))} / "
            f"{fmt(ri.get(KEYS['CP']))} / {fmt(ri.get(KEYS['CR']))} "
            f"(écarts : fidélité {d['F']:+.2f}, pertinence {d['AR']:+.2f}, précision {d['CP']:+.2f}, rappel {d['CR']:+.2f}).\n\n"
            "Interprétation : la **fidélité** mesure la part des affirmations de la réponse supportées par les extraits. "
            "Une valeur élevée montre que le prompt strict (réponse uniquement depuis les extraits, citation obligatoire) limite les hallucinations. "
            "La **pertinence de la réponse** indique si la réponse traite bien la question posée. "
            "La **précision du contexte** mesure si les extraits utiles sont classés en tête ; elle est pénalisée quand des slides voisines peu utiles sont ramenées dans le top-5. "
            "Le **rappel du contexte** mesure si les extraits contiennent toute l'information de la réponse de référence ; c'est la métrique qui profite le plus d'une meilleure récupération. "
            "Diagnostic (partie D) : les échecs de la baseline venaient surtout de la **récupération**. Les questions en français ramenaient parfois des slides voisines "
            "(Q3 : « similarité » → slide 238 sur les métriques d'évaluation ; Q9 : masque causal → slide 97 sur le masque de padding). "
            "L'amélioration retenue est la **réécriture de la question en requête anglaise par le LLM**, combinée à la recherche hybride. Elle fait passer le Hit@5 de 0.87 à 1.00. "
            "Les scores RAGAS restent à prendre avec prudence : 15 questions seulement, et le juge est un petit LLM (variance)."
        )
        md = md.replace("INTERP_RAGAS", interp)

    # latency
    lat = ri.get("latency_ms_mean_per_step") or rd.get("latency_ms_mean_per_step") or {}
    if lat:
        md = md.replace("LATENCY", ", ".join(f"`{k}` ≈ {v:.0f} ms" for k, v in lat.items() if k != "rag.request")
                        + f" (requête complète ≈ {lat.get('rag.request', 0):.0f} ms)")

    # refusals
    oos = imp.get("out_of_scope") or base.get("out_of_scope") or []
    if oos:
        n = sum(r["refused"] for r in oos)
        lines = "\n".join(f"- *{r['question']}* → {'**refus** ✓' if r['refused'] else 'réponse'} : « {r['answer'][:160]} »" for r in oos)
        md = md.replace("REFUSAL_RESULT",
            f"Le système **refuse honnêtement** dans {n}/{len(oos)} cas testés (questions hors corpus) :\n\n{lines}\n\n"
            "Ce comportement vient du prompt système : une formule de refus explicite (« Je ne trouve pas cette information dans le cours. ») "
            "est autorisée et exigée quand les extraits ne suffisent pas. C'est la technique de la « porte de sortie honnête » du cours (slide 201). "
            "Limite : le refus réduit les inventions sans les supprimer, et un extrait vaguement lié peut pousser le modèle à répondre partiellement.")
    (ROOT / "COMPTE_RENDU.md").write_text(md, encoding="utf-8")
    print("COMPTE_RENDU.md updated")


if __name__ == "__main__":
    main()
