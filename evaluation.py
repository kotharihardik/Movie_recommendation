"""
evaluate_engine.py
------------------
CineMatch India — Evaluation Metrics
=====================================
Offline evaluation for submission and TA defense.

This script uses proxy relevance labels because the project does not have
explicit user interaction logs. The goal is not to claim a production-grade
benchmark, but to report a transparent and defensible offline evaluation.

Primary ranking metric:
    - NDCG@10  (rank-aware, supports graded relevance)

Secondary ranking metrics:
    - Precision@K, Recall@K, MRR@K, MAP@K

Behavioral diagnostics:
    - ILD (intra-list diversity)
    - Fame bias (mean fame score of recommendations vs catalogue baseline)
    - Coverage (unique recommended items / catalogue size)

6 commonly used metrics in retrieval and recommendation literature:

    1. Precision@K   — fraction of top-K results that are relevant
    2. Recall@K      — fraction of all relevant items retrieved in top-K
    3. MRR@K         — reciprocal rank of the first relevant result
    4. MAP@K         — mean average precision (ranks all relevant items)
    5. NDCG@K        — rank-aware metric with graded relevance (gold standard)
    6. ILD           — Intra-List Diversity (pairwise cosine distance)

References:
    Weaviate Retrieval Evaluation Guide (2024) — weaviate.io/blog/retrieval-evaluation-metrics
    Shaped.ai — evaluating-recommendation-systems-map-mmr-ndcg
    arxiv 2312.16015 — Comprehensive Survey of Evaluation Techniques (2024)
    Kaminskas & Bridge, ACM TiiS 7(1), 2016 — diversity metric (ILD)

Practical note:
        - Because labels are unavailable, relevance is approximated from the movie
            metadata using genre overlap + keyword/cast/director overlap + vote quality.
        - This is appropriate for a course submission if you clearly state that
            the numbers are offline proxy metrics, not ground-truth user satisfaction.

Usage:
    from evaluate_engine import run_evaluation
    run_evaluation()                                        # default queries
    run_evaluation(queries=[("3 Idiots", ["hi"], "test")]) # custom
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pandas as pd


# ─── Terminal colours ────────────────────────────────────────────────────────
B = "\033[1m"; R = "\033[0m"; C = "\033[96m"; G = "\033[92m"
Y = "\033[93m"; RE = "\033[91m"; D = "\033[2m"; W = "\033[97m"

def _bar(v: float, w: int = 28) -> str:
    f = max(0, min(w, round(v * w)))
    c = G if v >= 0.65 else (Y if v >= 0.40 else RE)
    return f"{c}{'█'*f}{'░'*(w-f)}{R}"

def _grade(v: float) -> tuple[str, str]:
    if v >= 0.80: return "A", G
    if v >= 0.65: return "B", C
    if v >= 0.50: return "C", Y
    if v >= 0.35: return "D", Y
    return "F", RE

def _row(name: str, val: float, note: str = "") -> None:
    g, gc = _grade(val)
    note_s = f"  {D}{note}{R}" if note else ""
    print(f"  {W}{name:<22}{R}  {val:.4f}  {_bar(val)}  {gc}[{g}]{R}{note_s}")

def _hdr(title: str) -> None:
    print(f"\n{B}{C}{'━'*64}{R}")
    print(f"{B}{C}  {title}{R}")
    print(f"{B}{C}{'━'*64}{R}")

def _sec(letter: str, title: str) -> None:
    print(f"\n{B}  [{letter}] {title}{R}")
    print(f"  {'─'*56}")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for a metric list."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)], dtype=float)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


# ─── Ground-truth builder ────────────────────────────────────────────────────

def _genre_set(val) -> set:
    import re
    if isinstance(val, list):  return {str(g).strip().lower() for g in val if str(g).strip()}
    if isinstance(val, str):   return {g.strip().lower() for g in re.split(r"[|,;/]", val) if g.strip()}
    return set()


def _token_set(val) -> set[str]:
    if isinstance(val, list):
        return {str(v).strip().lower() for v in val if str(v).strip()}
    if isinstance(val, str):
        return {t.strip().lower() for t in val.replace("|", ",").split(",") if t.strip()}
    return set()


def _build_ground_truth(df: pd.DataFrame, anchor_idx: int) -> dict:
    """
    Offline proxy ground-truth.

    Because this project does not have explicit click/purchase logs, we build a
    graded relevance signal from metadata. This is not a substitute for real
    user labels, but it is defensible for a coursework submission as long as the
    limitation is stated clearly.

    Relevance heuristic:
        grade 2: strong genre + keyword/cast/director match, or strong genre overlap
        grade 1: weaker metadata overlap but still plausible similarity
    """
    anchor = df.loc[anchor_idx]
    ag     = _genre_set(anchor.get("genres", []))
    ak     = _token_set(anchor.get("keywords", []))
    ac     = _token_set(anchor.get("cast", []))
    ad     = str(anchor.get("director", "") or "").strip().lower()
    va     = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    vc     = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)

    rel, hi = set(), set()
    for idx, row in df.iterrows():
        if idx == anchor_idx or vc[idx] < 10:
            continue
        ov = len(ag & _genre_set(row.get("genres", [])))
        kw = len(ak & _token_set(row.get("keywords", [])))
        ca = len(ac & _token_set(row.get("cast", [])))
        dr = str(row.get("director", "") or "").strip().lower()
        r  = float(va[idx])

        grade = 0
        if ov >= 2 and r >= 6.5:
            grade = 2
        elif ov >= 1 and (kw >= 2 or ca >= 1 or dr == ad) and r >= 6.0:
            grade = 2
        elif ov >= 1 and (kw >= 1 or ca >= 1 or r >= 6.0):
            grade = 1

        if grade >= 1:
            rel.add(idx)
        if grade == 2:
            hi.add(idx)

    return {"relevant": rel, "highly_relevant": hi, "anchor_genres": ag, "anchor_keywords": ak, "anchor_cast": ac}


# ─── The 6 core metrics ──────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """P@K = |top-K ∩ relevant| / K"""
    return sum(1 for i in retrieved[:k] if i in relevant) / k if k else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """R@K = |top-K ∩ relevant| / |relevant|"""
    if not relevant: return 0.0
    return sum(1 for i in retrieved[:k] if i in relevant) / len(relevant)


def mrr_at_k(retrieved: list, relevant: set, k: int) -> float:
    """MRR@K = 1 / rank_of_first_relevant_result"""
    for rank, idx in enumerate(retrieved[:k], 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def map_at_k(retrieved: list, relevant: set, k: int) -> float:
    """MAP@K = (1/|R|) * sum_i P@i * rel(i)"""
    if not relevant: return 0.0
    hits, total = 0, 0.0
    for i, idx in enumerate(retrieved[:k], 1):
        if idx in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(retrieved: list, relevant: set, highly_relevant: set, k: int) -> float:
    """
    NDCG@K with graded relevance: highly_relevant=2, relevant=1, else=0.
    DCG@K = sum rel(i)/log2(i+1);  NDCG = DCG / IDCG
    """
    def rel(i): return 2 if i in highly_relevant else (1 if i in relevant else 0)
    dcg  = sum(rel(i) / math.log2(p + 2) for p, i in enumerate(retrieved[:k]))
    best = sorted([2]*len(highly_relevant) + [1]*len(relevant - highly_relevant), reverse=True)
    idcg = sum(r / math.log2(p + 2) for p, r in enumerate(best[:k]) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def ild(result_indices: list, embed_vecs: np.ndarray) -> float:
    """
    Intra-List Diversity: average pairwise cosine-distance within the result list.
    ILD = 0 means all items identical; ILD = 1 means maximally diverse.
    Reference: Kaminskas & Bridge (2016) ACM TiiS
    """
    idxs = [i for i in result_indices if i < len(embed_vecs)]
    if len(idxs) < 2: return 0.0
    vecs  = embed_vecs[idxs]          # already L2-normalised
    sims  = vecs @ vecs.T
    n     = len(idxs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([1.0 - float(sims[i, j]) for i, j in pairs]))


def _catalog_coverage(all_retrieved: list[list[int]], total_catalog_size: int) -> float:
    seen: set[int] = set()
    for rec in all_retrieved:
        seen.update(rec)
    if total_catalog_size <= 0:
        return 0.0
    return len(seen) / total_catalog_size


def _fame_bias(retrieved: list[int], fame_scores: np.ndarray) -> float:
    if fame_scores is None or len(retrieved) == 0:
        return 0.0
    idxs = [i for i in retrieved if i < len(fame_scores)]
    if not idxs:
        return 0.0
    return float(np.mean(fame_scores[idxs]))


def _coerce_str_list(value, limit: int | None = None) -> list[str]:
    import re

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[|,;/]", value)
    else:
        items = []

    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if limit is not None:
        return cleaned[:limit]
    return cleaned


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _is_clean_title(value) -> bool:
    title = str(value or "").strip()
    return bool(title) and title.lower() not in {"unknown", "na", "n/a", "none"}


def _build_eval_free_text(row: pd.Series) -> str:
    genres = _coerce_str_list(row.get("genres", []), limit=3)
    keywords = _coerce_str_list(row.get("keywords", []), limit=4)
    cast = _coerce_str_list(row.get("cast", []), limit=2)
    director = str(row.get("director", "") or "").strip()
    language = str(row.get("language", "") or "").strip()
    year = _safe_int(row.get("release_year", 0))

    genre_text = ", ".join(genres) if genres else "interesting"
    keyword_text = ", ".join(keywords) if keywords else "strong themes"
    cast_text = ", ".join(cast) if cast else "good performances"

    parts = [f"{language} {genre_text} movie".strip()]
    parts.append(f"with {keyword_text}")
    if director and director.lower() != "unknown":
        parts.append(f"directed by {director}")
    if cast:
        parts.append(f"starring {cast_text}")
    if year:
        parts.append(f"around {year}")

    return ", ".join(parts)


def _build_eval_description(row: pd.Series) -> str:
    genres = _coerce_str_list(row.get("genres", []), limit=2)
    language = str(row.get("language", "") or "").strip() or "mixed-language"
    year = _safe_int(row.get("release_year", 0))
    genre_text = "/".join(genres) if genres else "general"
    return f"{language} {genre_text} {year}".strip()


def _build_sampled_queries(
    df: pd.DataFrame,
    sample_size: int = 40,
    seed: int = 42,
    min_vote_count: int = 25,
    min_vote_average: float = 6.0,
) -> list[dict[str, object]]:
    """Build a stratified query set so evaluation is not based on a few hand-picked examples."""
    if df is None or len(df) == 0:
        return []

    eligible = df.copy()
    eligible = eligible[eligible["title"].apply(_is_clean_title)]
    eligible = eligible[pd.to_numeric(eligible["vote_count"], errors="coerce").fillna(0) >= min_vote_count]
    eligible = eligible[pd.to_numeric(eligible["vote_average"], errors="coerce").fillna(0) >= min_vote_average]
    eligible = eligible[eligible["language"].fillna("").astype(str).str.strip() != ""]

    if len(eligible) == 0:
        return []

    eligible = eligible.copy()
    vote_rank = pd.to_numeric(eligible["vote_count"], errors="coerce").fillna(0).rank(method="first")
    bucket_count = min(4, max(1, len(eligible)))
    try:
        eligible["pop_bucket"] = pd.qcut(vote_rank, q=bucket_count, labels=False, duplicates="drop")
    except ValueError:
        eligible["pop_bucket"] = 0

    rng = np.random.default_rng(seed)
    grouped_indices: list[list[int]] = []
    grouped_frames = eligible.groupby(["language", "pop_bucket"], dropna=False, sort=True)
    for _, group in grouped_frames:
        indices = list(group.index)
        rng.shuffle(indices)
        grouped_indices.append(indices)

    selected: list[int] = []
    pointers = [0] * len(grouped_indices)
    while len(selected) < sample_size:
        progressed = False
        for group_pos, indices in enumerate(grouped_indices):
            if pointers[group_pos] >= len(indices):
                continue
            selected.append(int(indices[pointers[group_pos]]))
            pointers[group_pos] += 1
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break

    if len(selected) < sample_size:
        remaining = [int(idx) for idx in eligible.index if int(idx) not in set(selected)]
        if remaining:
            fill_count = min(sample_size - len(selected), len(remaining))
            fill = rng.choice(remaining, size=fill_count, replace=False)
            selected.extend(int(idx) for idx in fill.tolist())

    queries: list[dict[str, object]] = []
    for rank, anchor_idx in enumerate(selected, 1):
        row = eligible.loc[anchor_idx]
        title = str(row.get("title", "") or "").strip()
        language = str(row.get("language", "") or "").strip()
        queries.append(
            {
                "anchor_idx": int(anchor_idx),
                "movie_title": title,
                "language_codes": [language] if language else [],
                "description": _build_eval_description(row),
                "free_text": _build_eval_free_text(row),
                "sample_rank": rank,
            }
        )

    return queries


# ─── Engine helpers ──────────────────────────────────────────────────────────

def _import_engine():
    try:
        import recommend_engine as eng
        return eng
    except ImportError as e:
        raise ImportError(
            "Cannot import recommend_engine.py — place both files in the same folder.\n" + str(e)
        )


def _map_to_df_indices(results, df: pd.DataFrame) -> list[int]:
    """Map RecommendedMovie objects to DataFrame row indices by title."""
    indices = []
    for r in results:
        m = df[df["title"] == r.title]
        if not m.empty:
            indices.append(int(m.index[0]))
    return indices


# ─── Default test queries ────────────────────────────────────────────────────
# (title, lang_codes, description)
DEFAULT_QUERIES = [
    ("3 Idiots",        ["hi"], "Hindi comedy-drama"),
    ("Dangal",          ["hi"], "Hindi sports-drama"),
    ("Sholay",          ["hi"], "Classic Bollywood"),
    ("Dil Chahta Hai",  ["hi"], "Comedy-drama"),
    ("Lagaan",          ["hi"], "Historical epic"),
    ("Kabir Singh",     ["hi"], "Romance-drama"),
]


def _normalize_query_entry(query, fallback_rank: int) -> dict[str, object]:
    if isinstance(query, dict):
        title = str(query.get("movie_title", "") or "").strip()
        lang_codes = list(query.get("language_codes") or [])
        desc = str(query.get("description", "") or title or f"Query {fallback_rank}").strip()
        free_text = str(query.get("free_text", "") or "").strip() or None
        anchor_idx = query.get("anchor_idx")
        return {
            "movie_title": title,
            "language_codes": lang_codes,
            "description": desc,
            "free_text": free_text,
            "anchor_idx": anchor_idx,
        }

    title, lang_codes, desc = query[:3]
    return {
        "movie_title": str(title),
        "language_codes": list(lang_codes),
        "description": str(desc),
        "free_text": None,
        "anchor_idx": None,
    }


# ─── Main evaluation runner ──────────────────────────────────────────────────

def run_evaluation(
    queries: Optional[list] = None,
    top_n: int = 10,
    min_rating: float = 5.0,
    ks: tuple[int, ...] = (5, 10, 20),
    bootstrap_iters: int = 1000,
    sample_size: int = 40,
    sample_seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run the core metrics over a set of queries and print results to terminal.

    Parameters
    ----------
    queries    : list of query tuples/dicts. If None, a stratified sample is built.
    top_n      : how many results to fetch per query
    min_rating : minimum vote_average filter passed to the engine
    ks         : ranking cutoffs to report
    sample_size: number of sampled anchors when queries is None
    sample_seed: seed used for stratified query sampling
    verbose    : print a more detailed per-query trace when True

    Returns
    -------
    dict of metric_name -> mean value across all queries
    """
    _hdr("CineMatch India — Evaluation Report")
    ks = tuple(sorted(set(int(k) for k in ks if int(k) > 0))) or (10,)
    primary_k = 10 if 10 in ks else ks[len(ks) // 2]
    print(f"  {D}Metrics : Precision@K  Recall@K  MRR@K  MAP@K  NDCG@K  ILD  Coverage  FameBias{R}")
    print(f"  {D}Primary : NDCG@{primary_k} (rank-aware)  |  Secondary: MAP@K, Recall@K, MRR@K{R}")
    print(f"  {D}Sources : Weaviate (2024)  |  Shaped.ai (2023)  |  arxiv 2312.16015{R}")

    eng    = _import_engine()
    if not eng._engine_ready:
        raise RuntimeError(
            "Engine not built. Call recommend_engine.build_engine(df) before run_evaluation()."
        )

    df         = eng._df
    embed_vecs = eng._embed_vecs
    fame_scores = getattr(eng, "_fame_scores", None)
    if queries is None:
        queries = _build_sampled_queries(
            df,
            sample_size=sample_size,
            seed=sample_seed,
            min_vote_count=25,
            min_vote_average=max(6.0, float(min_rating)),
        )
        if not queries:
            print(f"  {Y}Stratified sampling returned no queries; falling back to the small hand-written set.{R}")
            queries = [{
                "movie_title": title,
                "language_codes": lang_codes,
                "description": desc,
                "free_text": None,
                "anchor_idx": None,
            } for title, lang_codes, desc in DEFAULT_QUERIES]
        strategy_note = f"stratified sample of {len(queries)} anchors (seed={sample_seed})"
        query_note = "synthetic free-text built from metadata"
    else:
        strategy_note = f"manual query set ({len(queries)} queries)"
        query_note = "manual titles / optional free-text"

    print(f"  {D}Strategy : {strategy_note}{R}")
    print(f"  {D}Query type: {query_note}{R}")
    print(f"  {D}Sampling : min_vote_count>=25  min_vote_avg>=6.0  top_n={top_n}  seed={sample_seed}{R}")

    agg: dict[str, list[float]] = {f"Precision@{k}": [] for k in ks}
    agg.update({f"Recall@{k}": [] for k in ks})
    agg.update({f"MRR@{k}": [] for k in ks})
    agg.update({f"MAP@{k}": [] for k in ks})
    agg.update({f"NDCG@{k}": [] for k in ks})
    agg.update({"ILD": [], "FameBias": []})

    t0 = time.time()
    all_retrieved_lists: list[list[int]] = []
    all_fame_bias: list[float] = []
    evaluated_count = 0
    skipped_count = 0

    normalized_queries = [_normalize_query_entry(query, idx + 1) for idx, query in enumerate(queries)]

    for idx, query in enumerate(normalized_queries, 1):
        title = str(query["movie_title"])
        lang_codes = list(query["language_codes"])
        desc = str(query["description"])
        free_text = query.get("free_text")

        if verbose:
            print(f"\n  {B}{W}{desc}  ->  \"{title}\"{R}")

        try:
            results, _ = eng.get_recommendations(
                collection=None,
                movie_title=title,
                free_text=free_text,
                selected_chips=[],
                language_codes=lang_codes,
                top_n=top_n,
                min_rating=min_rating,
                decade_filter=None,
                include_old_movies=False,
                diversify=False,
                df=df,
            )
        except Exception as e:
            print(f"    {RE}x Failed: {e}{R}")
            skipped_count += 1
            continue

        if not results:
            if verbose:
                print(f"    {Y}! No results returned{R}")
            skipped_count += 1
            continue

        anchor_idx = eng._find_movie_idx(title)
        if anchor_idx is None:
            if verbose:
                print(f"    {Y}! Title not found in index{R}")
            skipped_count += 1
            continue

        retrieved = _map_to_df_indices(results, df)
        if not retrieved:
            if verbose:
                print(f"    {Y}! Could not map results to df indices{R}")
            skipped_count += 1
            continue

        gt              = _build_ground_truth(df, anchor_idx)
        relevant        = gt["relevant"]
        highly_relevant = gt["highly_relevant"]

        query_fame = _fame_bias(retrieved, fame_scores) - (float(np.mean(fame_scores)) if fame_scores is not None else 0.0)
        dv  = ild(retrieved, embed_vecs) if embed_vecs is not None else 0.0

        all_retrieved_lists.append(retrieved)
        all_fame_bias.append(query_fame)
        evaluated_count += 1

        per_k = {}
        for k in ks:
            p   = precision_at_k(retrieved, relevant, k)
            rc  = recall_at_k(retrieved, relevant, k)
            mr  = mrr_at_k(retrieved, relevant, k)
            ap  = map_at_k(retrieved, relevant, k)
            nd  = ndcg_at_k(retrieved, relevant, highly_relevant, k)
            per_k[k] = (p, rc, mr, ap, nd)
            agg[f"Precision@{k}"].append(p)
            agg[f"Recall@{k}"].append(rc)
            agg[f"MRR@{k}"].append(mr)
            agg[f"MAP@{k}"].append(ap)
            agg[f"NDCG@{k}"].append(nd)

        agg["ILD"].append(dv)
        agg["FameBias"].append(query_fame)

        p, rc, mr, ap, nd = per_k[primary_k]
        if verbose:
            print(f"    {D}relevant={len(relevant)}  highly_relevant={len(highly_relevant)}  "
                  f"retrieved={len(retrieved)}{R}")
            for k in ks:
                pk, rck, mrk, apk, ndk = per_k[k]
                _row(f"P@{k}", pk)
                _row(f"R@{k}", rck)
                if k == primary_k:
                    _row(f"MRR@{k}", mrk)
                    _row(f"MAP@{k}", apk)
                    _row(f"NDCG@{k}", ndk, "<- primary ranking metric")
            _row("ILD",            dv, "diversity within result list")
            _row("Fame bias",      query_fame, "recommendations vs catalogue baseline")
        else:
            print(
                f"  [{idx:02d}/{len(normalized_queries):02d}] "
                f"{title[:26]:<26} | {desc[:34]:<34} | "
                f"NDCG@{primary_k}={nd:.3f} R@{primary_k}={rc:.3f} "
                f"P@{primary_k}={p:.3f} ILD={dv:.3f} FameΔ={query_fame:+.3f}"
            )

    # ── Aggregate summary ────────────────────────────────────────────────────
    means = {k: _mean_std(v)[0] for k, v in agg.items() if v}

    _hdr(f"Mean Scores  ({evaluated_count} evaluated / {len(normalized_queries)} generated, K in {list(ks)})")
    print(f"  {D}Skipped : {skipped_count}{R}")

    _sec("Ranking", "Accuracy + Ranking Quality")
    for k in ks:
        _row(f"Precision@{k}", means.get(f"Precision@{k}", 0), f"accuracy of top-{k} list")
        _row(f"Recall@{k}",    means.get(f"Recall@{k}",    0), f"coverage of relevant items")
        if k == primary_k:
            _row(f"MRR@{k}",   means.get(f"MRR@{k}",       0), "rank of first relevant hit")
            _row(f"MAP@{k}",   means.get(f"MAP@{k}",       0), "avg precision over all relevant")
            _row(f"NDCG@{k}",  means.get(f"NDCG@{k}",      0), "graded rank-aware quality  <- primary")

    _sec("Diversity", "Intra-List Diversity  (Kaminskas & Bridge 2016)")
    _row("ILD",             means.get("ILD",       0), "0=identical  1=maximally diverse")

    _sec("Bias / Scope", "Coverage and Fame Bias")
    coverage = _catalog_coverage(all_retrieved_lists, len(df))
    fame_mean = float(np.mean(all_fame_bias)) if all_fame_bias else 0.0
    baseline_fame = float(np.mean(fame_scores)) if fame_scores is not None else 0.0
    _row("Coverage", coverage, "unique recommended items / catalogue size")
    _row("Fame bias", fame_mean, "positive means more popular movies than baseline")

    primary_scores = agg.get(f"NDCG@{primary_k}", [])
    ci_lo, ci_hi = _bootstrap_ci(primary_scores, n_boot=bootstrap_iters)
    recall_lo, recall_hi = _bootstrap_ci(agg.get(f"Recall@{primary_k}", []), n_boot=bootstrap_iters)

    core    = [means.get(f"NDCG@{primary_k}", 0), means.get(f"MAP@{primary_k}", 0), means.get(f"Recall@{primary_k}", 0)]
    overall = float(np.mean(core))
    g, gc   = _grade(overall)
    print(f"\n  {'─'*56}")
    print(f"  {B}Overall  {gc}{overall:.4f}  [{g}]{R}  {_bar(overall)}")
    print(f"  {D}NDCG@{primary_k} 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]{R}")
    print(f"  {D}Recall@{primary_k} 95% CI: [{recall_lo:.4f}, {recall_hi:.4f}]{R}")
    print(f"  {D}Baseline fame score: {baseline_fame:.4f}{R}")
    print(f"  {D}Elapsed: {time.time() - t0:.1f}s{R}")
    print(f"\n{B}{C}{'━'*64}{R}\n")

    return means


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from data_pipeline import run_full_pipeline
    from recommend_engine import build_engine

    DATA_PATH = os.environ.get("MOVIES_CSV", "data/movies.csv")
    DB_PATH   = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
    SAMPLE_SIZE = int(os.environ.get("EVAL_SAMPLE_SIZE", "40"))
    SAMPLE_SEED = int(os.environ.get("EVAL_SEED", "42"))
    VERBOSE = os.environ.get("EVAL_VERBOSE", "0") == "1"

    try:
        print(f"\n{D}Initializing engine for evaluation...{R}")
        # 1. Load data
        collection, df = run_full_pipeline(DATA_PATH, DB_PATH)
        
        # 2. Build engine
        build_engine(df)
        
        # 3. Run full evaluation
        run_evaluation(sample_size=SAMPLE_SIZE, sample_seed=SAMPLE_SEED, verbose=VERBOSE)

    except Exception as e:
        print(f"\n{RE}Error during evaluation: {e}{R}")
        print(f"{Y}Usage:{R}")
        print("  from evaluate_engine import run_evaluation")
        print("  run_evaluation()                                        # default")
        print("  run_evaluation(queries=[(\"3 Idiots\", [\"hi\"], \"test\")]) # custom\n")