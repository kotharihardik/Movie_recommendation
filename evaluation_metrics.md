# Evaluation metrics for CineMatch India

This file lists practical quantitative metrics you can use to evaluate the recommender offline. No single metric is "perfect" — combine a small set to measure accuracy (relevance), ranking quality, coverage, and bias.

Recommended primary metrics (top-N / ranking):

- Precision@k
  - What: fraction of recommended items in the top-k that are relevant.
  - Use when: you care about the immediate relevance of the top-k list.
  - Formula: Precision@k = |hits@k| / k

- Recall@k (aka Hit Rate@k)
  - What: fraction of all relevant items recovered in the top-k.
  - Use when: you care about coverage of ground-truth positives.
  - Formula: Recall@k = |hits@k| / |relevant|

- NDCG@k (Normalized Discounted Cumulative Gain)
  - What: measures ranking quality, giving higher weight to relevant items near the top.
  - Why: more reliable than raw precision when rank order matters.
  - Key formulas:
    - DCG@k = sum_{i=1..k} (2^{rel_i}-1) / log2(i+1)
    - IDCG@k = DCG@k for an ideal ranking
    - NDCG@k = DCG@k / IDCG@k
  - Use: set k (e.g., 10) and compute per-user then average.

- MAP@k (Mean Average Precision)
  - What: average precision across relevant items, then averaged over users.
  - Use when: you want precision that accounts for position of multiple relevant items.

- MRR (Mean Reciprocal Rank)
  - What: reciprocal of rank of first relevant item, averaged across queries/users.
  - Use when: you care mainly about placing at least one relevant hit near the top.

Secondary / distributional metrics (coverage, diversity, novelty, bias):

- Catalog Coverage
  - Fraction of unique items recommended across all users / total items.
  - Use: detect over-concentration on a small subset of movies.

- Aggregate Popularity / Fame Bias
  - Compare average `fame_score` in recommendations vs dataset baseline.
  - Option: report mean recommended fame and KL/divergence to catalogue fame distribution.
  - Use: quantify tendency to recommend popular films; useful for your "fame score" experiments.

- Diversity / Intra-list diversity
  - Compute average pairwise dissimilarity across items in each recommended list (e.g., 1 - cosine(sim) or 1 - Jaccard on genres) then average across users.
  - Use: avoid near-duplicate lists and improve user discovery.

- Novelty / Serendipity
  - Novelty: measure average popularity rank or self-information (-log p(item)); higher is more novel.
  - Serendipity: quantify unexpectedness relative to user history (requires user profile) — often approximated as novelty weighted by user interest match.

Evaluation protocol (recommended):

1. Use a temporal holdout when possible (train on t < T, test on interactions after T) to simulate real recommendation flow.
2. For top-N evaluation, withhold one or more interactions per user as ground-truth (leave-one-out or leave-two-out) and rank items among all candidates.
3. Evaluate per-user metrics and report macro-averaged mean +/- standard deviation.
4. Report metrics at multiple cutoffs: @5, @10, @20.
5. Always accompany accuracy metrics with coverage and popularity/fame bias metrics.

Practical notes and references

- For list recommendation, use NDCG@k and MAP@k as primary ranking metrics; report Recall@k/Precision@k for interpretability.
- For production-facing systems, include coverage and fame-popularity bias diagnostics.
- Common references: RecSys tutorials and standard recommender evaluation literature (e.g., "Evaluating Recommender Systems" surveys and RecSys conference papers).

Next steps

- See `evaluation.py` for a small runnable harness that computes Precision@k, Recall@k, AP@k, MAP@k, NDCG@k, MRR, and Coverage from recommended lists and ground-truth sets.
- You can extend `evaluation.py` with diversity/novelty functions once item embeddings or popularity statistics are available.
