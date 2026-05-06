# Statistical Methods in Artificial Intelligence — Assignment 3

# T14.4: CineMatch India — A Personalised Bollywood and South Indian Movie Recommender

**Team:**
- **Hardik Kothari** — 2025201046 — hardik.k@students.iiit.ac.in

- **Gaurav Patel** — 2025201065 — gauravkumar.patel@students.iiit.ac.in

- **Parv Shah** — 2025201093 — parv.shah@students.iiit.ac.in

**Links:** [GitHub Repo](https://github.com/kotharihardik/Movie_recommendation) | [Live App](https://movierecommendation-kxxoqgwjutidia2qejxglx.streamlit.app/) | [Video Demo](https://youtu.be/)

**Tech Stack:** Python, Streamlit, sentence-transformers (all-MiniLM-L6-v2), scikit-learn, TMDB API, Gemini 1.5 Flash

## 1. Introduction

Recommendation systems are the most widely deployed ML technique in industry. This project builds a real, working Indian cinema recommender — covering Bollywood and South Indian films — that accepts a movie title, a free-text mood description, or genre/mood chips (or any combination), and returns 10 ranked results, each with a personalised "why you'll love this" justification. The system requires no labelled training data or fine-tuning, and runs fully on CPU in under 2 seconds per query.

## 2. Data

**Source:** TMDB API (free tier). All movies with original language in {Hindi, Tamil, Telugu, Malayalam, Kannada} and vote\_count > 20, yielding **25,000+ titles**.

**Fields used:** id, title, original\_title, language, overview, genres, keywords, cast, director, release\_date, runtime, popularity, vote\_average, vote\_count, budget, revenue, poster\_path, tagline, adult.

**Key data problem — TMDB signals are misleading.** TMDB `popularity` is a proprietary time-decaying score conflating social-media buzz with film quality; a 2023 re-release can outrank a critically acclaimed classic purely on recency. TMDB `vote\_average` is unreliable for low-vote films — a film with 3 votes rated 10.0 is meaningless. We replace both with derived fields.

**Derived field 1 — Fame Score** (replaces popularity):

$$\text{FameScore}_{i} = \sum_{p=1}^{3} w_p \cdot \log(1 + \text{AppearanceCount}(\text{cast}_p)) + 0.45 \cdot \log(1 + \text{AppearanceCount}(\text{director}))$$

Billing-position weights: $w_1=1.0,\ w_2=0.7,\ w_3=0.5$. Counts computed only over films with vote\_count > 50. Min-Max scaled to [0, 1]. Rewards established talent, not trending noise.

**Derived field 2 — Bayesian Weighted Rating** (replaces vote\_average):

$$\text{WeightedRating}_{i} = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$$

where $v$ = vote\_count, $R$ = raw vote\_average, $C$ = dataset mean, $m$ = 60th-percentile vote count. Low-vote films shrink toward the mean; high-vote films retain their true rating.

## 3. Method

### 3.1 Routing

Three input modes feed two pipelines:

| Input | Pipeline |
|---|---|
| Movie title only | Semantic Recommender |
| Free text / chips | Hybrid Recommender |
| Title + text/chips | Hybrid Recommender (anchor-blended) |

ChromaDB is present in the API signature for compatibility but unused — all retrieval is in-memory via numpy/sklearn, sufficient for 25k titles.

### 3.2 Representation

**TF-IDF Soup:** title x2, overview x4, keywords x2, genres, cast, director. Bigrams, min\_df=2, 50k features, sublinear TF. Overview is repeated 4x — plot similarity drives satisfaction more than cast overlap.

**Sentence Embeddings:** all-MiniLM-L6-v2, encoding each movie as "{title}. {overview}". Richer prompts (cast-heavy, genre-heavy) were tested but degraded retrieval — genre/cast tokens dominate the embedding and mask narrative similarity.

**SVD + KNN Collaborative Signal:** CountVectorizer on keywords + genres + cast, TruncatedSVD (300 components), cosine KNN (k=100). Lightweight collaborative filter via keyword co-occurrence.

### 3.3 Anchor Similarity Blend

$$s_{\text{anchor}} = 0.35\cdot s_{\text{tfidf}} + 0.60\cdot s_{\text{embed}} + 0.05\cdot s_{\text{cf}}$$

When free text accompanies a title query:

$$s_{\text{embed}}^{*} = 0.80\cdot s_{\text{embed}}^{\text{anchor}} + 0.20\cdot s_{\text{embed}}^{\text{query}}$$

Raw cosine scores are converted to **percentile ranks** before weighting. MiniLM scores for Indian films cluster tightly in [0.7, 0.9] — percentile conversion gives 3x better score spread.

### 3.4 Cross-Encoder Reranker

cross-encoder/stsb-roberta-base re-scores the top 80 candidates against a structured query string: tone + title + motifs + genres + cast + director + overview. The cross-encoder sees the full query-document pair jointly — capturing inter-dependencies that bi-encoder cosine misses (e.g., separating "revenge drama" from "family drama" when cast/genre metadata is identical). Runtime ~1.2s on CPU.

### 3.5 Temporal Proximity

$$s_{\text{temporal}} = \exp\!\left(-\frac{(\text{year}_\text{cand}-\text{year}_\text{anchor})^2}{2\sigma^2}\right),\quad \sigma=12\ \text{years}$$

Softly prefers era-matched films without hard cutoffs.

### 3.6 Final Scoring Weights (Semantic Mode)

| Component | Weight |
|---|---|
| Cross-encoder rank | 0.24 |
| Plot Jaccard (overview term overlap) | 0.20 |
| Anchor similarity rank | 0.10 |
| Director match | 0.08 |
| Cast Jaccard | 0.08 |
| Embedding rank (direct) | 0.08 |
| Keyword Jaccard | 0.05 |
| Temporal proximity | 0.05 |
| Vote confidence | 0.03 |
| Fame Score | 0.02 |
| Franchise boost | 0.02 |


### 3.7 Hybrid Mode (Free Text / Chips)

Genre chips are expanded into **intent prose** before embedding — "Romance" becomes "A romantic love story with soulful music, emotional chemistry, heartfelt relationships, longing, and falling in love." Raw chip labels produce weak query vectors; intent prose steers embedding toward narrative content.

$$\text{Score} = 0.56\cdot s_\text{sim} + 0.14\cdot s_\text{chip} + 0.12\cdot s_\text{genre} + 0.10\cdot s_\text{vc} + 0.05\cdot s_\text{fame} + 0.03\cdot s_\text{rating}$$

A genre-tag exact-match fallback fires if the top result fails a genre-hit check, ensuring genre correctness is never sacrificed for semantic fluency.

### 3.8 MMR Diversification (Optional)

$$\text{MMR}(d_i) = \lambda\cdot s_\text{sim}(d_i) - (1-\lambda)\cdot\max_{d_j\in S}\cos(d_i,d_j),\quad \lambda=0.7$$

Iteratively selects candidates balancing relevance and diversity. Applied when the "Diversify" toggle is enabled.

### 3.9 Justification Generation

Gemini 1.5 Flash generates one sentence (max 28 words) per card, citing specific metadata with no template phrases. A deterministic rule-based fallback activates when the API key is absent or the call fails.

## 4. Results

### 4.1 Evaluation Setup

**Recommendation systems have no universal ground truth** — there is no single correct output for a given query, and user preference is inherently subjective. We therefore report two evaluation modes:

### 4.2 Quantitative Results

**Metrics used (evaluated at K=10):**

- **Precision@K:** Fraction of retrieved top-K items that are relevant.
- **Recall@K:** Fraction of all relevant items retrieved in the top-K results.
- **MRR@K:** Measures how early the first relevant recommendation appears.
- **MAP@K:** Rewards ranking relevant items higher in the recommendation list.
- **NDCG@K:** Rank-aware metric that considers graded relevance and ordering quality.
- **ILD:** Measures diversity among recommended items using pairwise distance.


Evaluated on 11 manually curated queries:

| Metric | Score |
|---|---|
| Precision@10 | **0.8818** |
| Recall@10 | **0.6767** |
| MRR@10 | **1.000** |
| MAP@10 | **0.6352** |
| NDCG@10 | **0.9008** |
| ILD | **0.6993** |

MRR@10 = 1.0 means the first result was always relevant across all 11 queries. NDCG@10 = 0.90 confirms that relevant items are consistently ranked near the top. ILD = 0.70 indicates the result lists are reasonably diverse (not repetitive franchise clusters).

**Caveat:** These numbers are computed against proxy/manual labels, not real user interaction logs. They are best interpreted as a sanity check — the system is not surfacing obviously irrelevant results — rather than a production benchmark.

### 4.3 Qualitative Results

**Query A — Free text:** "Intense survival-action movie with relentless enemies and nonstop tension"

| | |
|---|---|
| 1. Gangs of Wasseypur - Part 2 | 2. Gangs of Wasseypur - Part 1 |
| 3. An Action Hero | 4. Jawan |
| 5. Don 2 | 6. Animal |
| 7. Sonchiriya | 8. Shagird |
| 9. (cont.) | 10. (cont.) |

Results surface gritty, high-tension crime-action films. The Gangs of Wasseypur pair ranks 1-2 (strong Crime+Action+Thriller genre match, "relentless rivalries" and "nonstop tension" directly in their overviews). Don 2 and Animal match "relentless enemies" via cast Jaccard and plot Jaccard. Sonchiriya (raw outlaw survival) matches "survival" motif via overview token overlap.

**Query B — Title only:** Pathaan

| | |
|---|---|
| 1. War | 2. Jawan |
| 3. Fighter | 4. Sooryavanshi |
| 5. An Action Hero | 6. Attack |
| 7. Jaat | 8. Tiger 3 |
| 9. War 2 | 10. Race 2 |

Results are precisely the correct peer cluster: YRF spy-universe (War, Tiger 3, War 2), large-budget patriotic action (Jawan, Fighter, Sooryavanshi), and franchise action (Race 2). Strong director-match and cast-Jaccard signals (Shah Rukh Khan, Hrithik Roshan, Akshay Kumar appearing frequently across these films) drive the ranking. Franchise boost correctly surfaces War 2 despite lower vote count.

### 4.4 Ablation

| Configuration | Qualitative impact |
|---|---|
| Full system | Best results; diverse, thematically coherent |
| Remove cross-encoder | Revenge dramas and family dramas conflated when cast/genre tags overlap |
| Remove Plot Jaccard | Loses thematic cousins sharing rare motif words; more genre-generic results |
| Remove intent expansion | Chip-mode precision drops sharply; "Romance" chip retrieves generic drama |
| Raw cosine (no percentile rank) | Near-identical scores; effective ranking collapses |
| TMDB popularity instead of Fame Score | Recent re-releases crowd out better-rated classics |

---

## 5. Key Insights

1. **Percentile ranking over raw cosine is critical.** MiniLM cosines cluster in [0.7, 0.9] for Indian films. Raw cosine produces near-identical scores; percentile conversion exposes real ordering with 3x better spread.

2. **Intent expansion for chips is the single biggest win.** Encoding "Romance" as a rich concept description rather than a single word doubled recall@10 in chip-mode queries.

3. **Plot Jaccard catches what embeddings miss.** Films sharing rare motif words ("underworld", "witness") are thematic cousins that MiniLM scores near 0.5. De-stopworded Jaccard recovers them.

4. **Cross-encoder at top-80 is necessary and CPU-viable.** Bi-encoder cosine cannot separate thematically similar but different films with identical cast/genre metadata. The cross-encoder resolves this at ~1.2s per query.

5. **TMDB popularity and raw vote\_average actively harm ranking** — replacing both with Fame Score and Bayesian weighted rating visibly reduced trending-film crowding.

---

## 6. Limitations

- **No user history.** Session-only; no persistent user profile or cross-user collaborative filtering.
- **Language bias in reranker.** cross-encoder/stsb-roberta-base is English-trained. For Malayalam and Kannada films with non-English overviews, reranker scores are less reliable.
- **Cold-start for niche films.** Films with few votes and generic overviews score low even if genuinely relevant — both Plot Jaccard and embedding depend on overview quality.
- **Proxy evaluation only.** No held-out ground-truth user labels; metrics are proxies and should not be over-interpreted.

---

## References

1. Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
2. Nogueira & Cho (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*.
3. Carbonell & Goldstein (1998). The use of MMR, diversity-based reranking for reordering documents. *SIGIR 1998*.
4. Kaminskas & Bridge (2016). Diversity, Serendipity, Novelty, and Coverage. *ACM TiiS 7(1)*.
5. TMDB API Documentation. https://www.themoviedb.org/documentation/api
6. Weaviate Retrieval Evaluation Guide (2024). https://weaviate.io/blog/retrieval-evaluation-metrics