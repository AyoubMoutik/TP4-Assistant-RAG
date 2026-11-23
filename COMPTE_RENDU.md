# COMPTE-RENDU — TP 4 : Assistant documentaire (RAG) avec évaluation & traçage

## Identité & configuration

- **Nom(s) et prénom(s) :** OUZZANE Jamila · MOUTIK Ayoub · ELKHALI Omar  **Filière / groupe :** BDIO — S3
- **Cas d'usage :** 1 — Assistant du cours (PDF du module « Transformers & LLMs », 252 slides, 6 cours, figures incluses)
- **Langue :** FR ☑ AR ☐ — questions, réponses, prompts et golden set en français ; le corpus fourni est en anglais → recherche **cross-lingue** FR → EN
- **LLM :** Gemini `gemini-3.5-flash-lite` (génération) ; juge RAGAS : `gemini-3.1-flash-lite` (modèle distinct)
- **Voie :** API ☑ (clé Gemini gratuite) Local (Ollama) ☐
- **Embedding :** `intfloat/multilingual-e5-base` (local, multilingue)
- **Chunking :** 1 slide = 1 chunk (découpage si > 1500 car.) + en-tête « cours | section | titre » + description des figures
- **top-k :** 5
- **Base vectorielle :** FAISS (similarité cosinus)

## Scores RAGAS (golden set : 15 questions)

| Métrique | Fidélité | Pertinence réponse | Précision contexte | Rappel contexte |
|---|---|---|---|---|
| Baseline (dense, question FR) | 1.00 | 0.91 | 0.91 | 0.93 |
| **Amélioré (réécriture FR→EN + hybride)** | 1.00 | 0.91 | 0.90 | 1.00 |

Métriques de récupération complémentaires (sans LLM juge, pages de référence du golden set) :

| Configuration | Hit@5 | MRR |
|---|---|---|
| dense (baseline) | 0.87 | 0.800 |
| hybride dense + BM25 (RRF) | 0.87 | 0.661 |
| dense + réécriture FR→EN | 0.93 | 0.813 |
| **hybride + réécriture FR→EN** | **1.00** | **0.856** |

## Exemple Q/R avec citations

**Question :** Pourquoi divise-t-on les scores d'attention par la racine carrée de d_k ?

**Réponse :** On divise les scores d'attention par √d_k pour stabiliser les scores lorsque la dimension d_k devient grande, car des scores trop élevés en valeur absolue font saturer la fonction softmax [1][4]. Si les composantes de Q et K ont une variance de 1, leur produit scalaire a une variance de d_k, et diviser par √d_k ramène cette variance à 1 [1]. Cela évite que la distribution du softmax ne devienne presque binaire, ce qui écraserait les gradients et bloquerait l'apprentissage par rétropropagation [4].

**Sources citées :**
- [1] Lecture 2A — The Attention Mechanism, p. 55 — « Dividing by √dₖ: stabilizing the scores »
- [4] Lecture 2A — The Attention Mechanism, p. 54 — « ❓ Why can scores be problematic? »

## Réponses aux questions

**Q1. Stratégie de chunking ? Pourquoi la structure compte ?**
Le corpus est un support de cours en slides : l'unité logique est la **slide** (équivalent de l'« article » d'un texte juridique). Chaque slide forme un chunk (les rares slides > 1500 caractères sont découpées avec chevauchement de 150 caractères, pour rester sous les 512 tokens d'e5). Chaque chunk est préfixé par un en-tête structurel « cours | section | titre de slide » et porte des métadonnées : source, n° de page, cours, section, titre, présence de figure, langue. Les 49 slides contenant des figures reçoivent une description générée par Gemini vision, ajoutée au texte (« y compris les figures »). Résultat : 275 chunks. Respecter la structure évite de couper une définition ou une formule en deux, garde chaque chunk autonome et permet une citation précise (cours + page). C'est encore plus vrai en juridique : un article est l'unité de sens *et* l'unité de citation ; un chunk qui mélange deux articles produit des réponses ambiguës et des citations fausses.

**Q2. L'embedding est-il multilingue ? Pourquoi est-ce indispensable pour l'arabe ?**
Oui : `multilingual-e5-base` est entraîné sur 100+ langues (dont le français et l'arabe) et projette des phrases de langues différentes dans **le même espace vectoriel**. C'est ce qui rend notre système possible : les questions sont en français et les slides en anglais. Un modèle anglais seul tokeniserait mal l'arabe (alphabet, morphologie riche, mots en sous-tokens absurdes) et ne rapprocherait pas une question arabe d'un passage équivalent. Limite observée : le cross-lingue direct reste imparfait (Hit@5 = 0.87). Par exemple, « similarité entre deux embeddings » ramène d'abord la slide sur les métriques de similarité d'évaluation. Cela a motivé l'amélioration de la partie D.

**Q3. Comment le système cite-t-il ses sources ? Pourquoi est-ce essentiel ?**
Les 5 extraits sont numérotés [1]…[5] dans le prompt avec leur cours, n° de slide et titre. Le prompt système impose de citer le numéro de l'extrait après chaque phrase factuelle. Le code extrait ensuite les numéros cités et affiche la liste des sources : `[1] Lecture 2A — The Attention Mechanism, p. 55 — « Dividing by √dₖ… »`. L'étudiant peut ainsi vérifier sur la slide exacte. En juridique, c'est essentiel : une réponse n'a de valeur que si elle renvoie à l'article de loi précis (vérifiabilité, opposabilité, responsabilité). Cela permet aussi de détecter une hallucination : une affirmation sans citation, ou une citation qui ne la contient pas.

**Q4. Scores RAGAS et interprétation.**
Baseline : fidélité 1.00, pertinence 0.91, précision du contexte 0.91, rappel du contexte 0.93. Version améliorée (réécriture FR→EN + hybride) : 1.00 / 0.91 / 0.90 / 1.00 (écarts : fidélité +0.00, pertinence -0.00, précision -0.01, rappel +0.07).

Interprétation : la **fidélité** mesure la part des affirmations de la réponse supportées par les extraits. Une valeur élevée montre que le prompt strict (réponse uniquement depuis les extraits, citation obligatoire) limite les hallucinations. La **pertinence de la réponse** indique si la réponse traite bien la question posée. La **précision du contexte** mesure si les extraits utiles sont classés en tête ; elle est pénalisée quand des slides voisines peu utiles sont ramenées dans le top-5. Le **rappel du contexte** mesure si les extraits contiennent toute l'information de la réponse de référence ; c'est la métrique qui profite le plus d'une meilleure récupération. Diagnostic (partie D) : les échecs de la baseline venaient surtout de la **récupération**. Les questions en français ramenaient parfois des slides voisines (Q3 : « similarité » → slide 238 sur les métriques d'évaluation ; Q9 : masque causal → slide 97 sur le masque de padding). L'amélioration retenue est la **réécriture de la question en requête anglaise par le LLM**, combinée à la recherche hybride. Elle fait passer le Hit@5 de 0.87 à 1.00. Les scores RAGAS restent à prendre avec prudence : 15 questions seulement, et le juge est un petit LLM (variance).

**Q5. Fidélité basse MAIS rappel du contexte élevé : récupération ou génération ?**
C'est un problème de **génération**. Un rappel du contexte élevé signifie que les bons passages ont bien été récupérés : l'information nécessaire était dans le prompt. Une fidélité basse signifie que la réponse contient des affirmations non supportées par ces passages : le LLM a ajouté des connaissances externes, mal paraphrasé ou extrapolé. Remèdes : prompt plus strict (« uniquement à partir des extraits », citation obligatoire), température 0, modèle plus fort, contexte moins bruité (moins de chunks ou reranking), vérification post-génération des citations. Dans notre cas, la fidélité est de 1.00 sur les deux configurations : le prompt strict suffit, et nos échecs étaient côté récupération (rappel 0.93 → 1.00).

**Q6. Contexte insuffisant : invente-t-il ou refuse-t-il ?**
Le système **refuse honnêtement** dans 3/3 cas testés (questions hors corpus) :

- *Quelle est la durée légale du congé de maternité au Maroc selon le Code du travail ?* → **refus** ✓ : « Je ne trouve pas cette information dans le cours. »
- *Combien de paramètres possède exactement le modèle Gemini 2.5 Pro ?* → **refus** ✓ : « Je ne trouve pas cette information dans le cours. [2] »
- *Comment configurer un cluster Kubernetes pour servir un LLM avec vLLM ?* → **refus** ✓ : « Je ne trouve pas cette information dans le cours. »

Ce comportement vient du prompt système : une formule de refus explicite (« Je ne trouve pas cette information dans le cours. ») est autorisée et exigée quand les extraits ne suffisent pas. C'est la technique de la « porte de sortie honnête » du cours (slide 201). Limite : le refus réduit les inventions sans les supprimer, et un extrait vaguement lié peut pousser le modèle à répondre partiellement.

**Q7. (Bonus) Hybride / reranking : gain mesuré ? Traçage : étape la plus lente ?**
- **Hybride (dense + BM25, fusion RRF)** appliqué seul sur la question française : aucun gain, et même une dégradation (MRR 0.800 → 0.661). BM25 est lexical, et les mots français ne correspondent pas aux slides anglaises.
- **Combiné à la réécriture FR→EN**, l'hybride devient la meilleure configuration : Hit@5 = 1.00, MRR = 0.856 (+13 points de Hit@5 par rapport à la baseline). Les termes techniques exacts (« RMSNorm », « causal mask ») sont alors bien capturés par BM25.
- **Reranking :** un cross-encoder multilingue (`BAAI/bge-reranker-base`) est implémenté (`--rerank`) mais n'a pas été mesuré, faute de temps.
- **Traçage OpenTelemetry :** un span par étape (`rag.query_rewrite`, `rag.retrieve`, `rag.rerank`, `rag.generate`). Latences moyennes : `rag.query_rewrite` ≈ 0 ms (réponses servies depuis le cache disque ; sans cache c'est un appel LLM d'environ 1 s), `rag.retrieve` ≈ 62 ms, `rag.generate` ≈ 3871 ms (requête complète ≈ 3934 ms). L'étape la plus lente est la **génération LLM** (appel réseau + décodage token par token). La réécriture ajoute un second appel LLM. La recherche FAISS + BM25 est négligeable (quelques dizaines de ms).

**Q8. API vs local pour un cabinet soumis à la CNDP ?**
Recommandation : **local**, ou serveur auto-hébergé au Maroc. La loi 09-08 et la CNDP encadrent le traitement des données personnelles et leur **transfert à l'étranger**, qui demande une autorisation ou des garanties. Une API cloud (Gemini, OpenAI…) envoie les documents des clients sur des serveurs hors du Maroc, ce qui pose un problème de transfert, de secret professionnel et de rétention par le fournisseur. Un modèle open-weight (Gemma, Qwen, Mistral, LLaMA) servi via Ollama (poste) ou vLLM (serveur) garde les données dans le réseau du cabinet. Notre pipeline le permet déjà : les embeddings, FAISS, BM25 et le reranker tournent en local, seul le LLM est à remplacer. Le cours le rappelle (slide 188) : auto-héberger n'est pas automatiquement conforme. Il faut aussi le contrôle d'accès, la journalisation, le chiffrement, l'anonymisation et la déclaration du traitement à la CNDP. L'API reste acceptable pour un prototype sur données publiques ou anonymisées.

## Décision (API vs local) & CNDP

Pour ce TP, avec un corpus public (le cours) : API Gemini gratuite, adaptée au prototypage. Pour un cabinet juridique soumis à la CNDP : déploiement **local** (Ollama/vLLM + modèle open-weight), avec la même architecture (embedding multilingue local + FAISS + citations). Il faut aussi anonymiser les documents avant indexation et déclarer le traitement à la CNDP. Le coût est matériel (GPU) et la qualité un peu inférieure, mais c'est le seul moyen de garder la maîtrise des données.

## Limites & pistes d'amélioration

- **Quotas de l'API gratuite :** 20 requêtes/jour pour `gemini-2.5-flash`. Nous avons dû utiliser des modèles « lite », moins bons, pour la génération et le juge RAGAS, d'où des scores RAGAS bruités (le juge est lui-même un LLM).
- **Golden set petit** (15 questions) et rédigé par nous : il faudrait 50+ questions, dont des questions multi-slides et des questions sur les figures.
- **Corpus en anglais :** la recherche cross-lingue directe rate certaines questions. La réécriture FR→EN corrige la plupart des cas mais coûte un appel LLM de plus. Alternative : indexer une traduction française des slides.
- **Figures :** décrites par un VLM (risque d'erreur dans les descriptions) ; les formules extraites du PDF sont parfois mal linéarisées.
- **Pistes :** mesurer le reranking, utiliser un embedding plus grand (`bge-m3`), faire tourner le LLM en local (Ollama), vérifier automatiquement les citations, et ajouter un RAG conversationnel évalué (déjà implémenté : `--chat`).
