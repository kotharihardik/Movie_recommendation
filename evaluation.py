"""
evaluate_engine.py
------------------
CineMatch India — Evaluation Metrics
=====================================
6 industry-standard metrics used on MovieLens benchmarks:

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


# ─── Ground-truth builder ────────────────────────────────────────────────────

def _genre_set(val) -> set:
    import re
    if isinstance(val, list):  return {str(g).strip().lower() for g in val if str(g).strip()}
    if isinstance(val, str):   return {g.strip().lower() for g in re.split(r"[|,;/]", val) if g.strip()}
    return set()


def _build_ground_truth(df: pd.DataFrame, anchor_idx: int) -> dict:
    """
    Offline pseudo ground-truth (standard practice when explicit user ratings
    are unavailable — used in MovieLens benchmark papers):

        relevant        = genre overlap >= 1  AND  vote_average >= 6.0
        highly_relevant = genre overlap >= 2  AND  vote_average >= 7.0
    """
    anchor = df.loc[anchor_idx]
    ag     = _genre_set(anchor.get("genres", []))
    va     = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    vc     = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)

    rel, hi = set(), set()
    for idx, row in df.iterrows():
        if idx == anchor_idx or vc[idx] < 10:
            continue
        ov = len(ag & _genre_set(row.get("genres", [])))
        r  = float(va[idx])
        if ov >= 1 and r >= 6.0: rel.add(idx)
        if ov >= 2 and r >= 7.0: hi.add(idx)

    return {"relevant": rel, "highly_relevant": hi, "anchor_genres": ag}


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

K = 10   # standard cut-off used in MovieLens benchmarks


# ─── Main evaluation runner ──────────────────────────────────────────────────

def run_evaluation(
    queries: Optional[list] = None,
    top_n: int = 10,
    min_rating: float = 5.0,
) -> dict[str, float]:
    """
    Run the 6 core metrics over a set of queries and print results to terminal.

    Parameters
    ----------
    queries    : list of (title, lang_codes, description) tuples
                 defaults to DEFAULT_QUERIES if None
    top_n      : how many results to fetch per query
    min_rating : minimum vote_average filter passed to the engine

    Returns
    -------
    dict of metric_name -> mean value across all queries
    """
    _hdr("CineMatch India — Evaluation Report")
    print(f"  {D}Metrics : Precision@{K}  Recall@{K}  MRR@{K}  MAP@{K}  NDCG@{K}  ILD{R}")
    print(f"  {D}Sources : Weaviate (2024)  |  Shaped.ai (2024)  |  arxiv 2312.16015{R}")

    eng    = _import_engine()
    if not eng._engine_ready:
        raise RuntimeError(
            "Engine not built. Call recommend_engine.build_engine(df) before run_evaluation()."
        )

    df         = eng._df
    embed_vecs = eng._embed_vecs
    queries    = queries or DEFAULT_QUERIES

    agg: dict[str, list[float]] = {
        "Precision": [], "Recall": [], "MRR": [], "MAP": [], "NDCG": [], "ILD": []
    }

    t0 = time.time()

    for title, lang_codes, desc in queries:
        print(f"\n  {B}{W}{desc}  ->  \"{title}\"{R}")

        try:
            results, _ = eng.get_recommendations(
                collection=None,
                movie_title=title,
                free_text=None,
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
            continue

        if not results:
            print(f"    {Y}! No results returned{R}")
            continue

        anchor_idx = eng._find_movie_idx(title)
        if anchor_idx is None:
            print(f"    {Y}! Title not found in index{R}")
            continue

        retrieved = _map_to_df_indices(results, df)
        if not retrieved:
            print(f"    {Y}! Could not map results to df indices{R}")
            continue

        gt              = _build_ground_truth(df, anchor_idx)
        relevant        = gt["relevant"]
        highly_relevant = gt["highly_relevant"]

        p   = precision_at_k(retrieved, relevant, K)
        rc  = recall_at_k(retrieved, relevant, K)
        mrr = mrr_at_k(retrieved, relevant, K)
        ap  = map_at_k(retrieved, relevant, K)
        n   = ndcg_at_k(retrieved, relevant, highly_relevant, K)
        dv  = ild(retrieved, embed_vecs) if embed_vecs is not None else 0.0

        agg["Precision"].append(p)
        agg["Recall"].append(rc)
        agg["MRR"].append(mrr)
        agg["MAP"].append(ap)
        agg["NDCG"].append(n)
        agg["ILD"].append(dv)

        print(f"    {D}relevant={len(relevant)}  highly_relevant={len(highly_relevant)}  "
              f"retrieved={len(retrieved)}{R}")
        _row(f"Precision@{K}", p)
        _row(f"Recall@{K}",    rc)
        _row(f"MRR@{K}",       mrr)
        _row(f"MAP@{K}",       ap)
        _row(f"NDCG@{K}",      n,  "<- primary ranking metric")
        _row("ILD",            dv, "diversity within result list")

    # ── Aggregate summary ────────────────────────────────────────────────────
    means = {k: float(np.mean(v)) for k, v in agg.items() if v}

    _hdr(f"Mean Scores  ({len(queries)} queries, K={K})")

    _sec("Ranking", "Accuracy + Ranking Quality")
    _row(f"Precision@{K}",  means.get("Precision", 0), "accuracy of top-K list")
    _row(f"Recall@{K}",     means.get("Recall",    0), "coverage of relevant items")
    _row(f"MRR@{K}",        means.get("MRR",       0), "rank of first relevant hit")
    _row(f"MAP@{K}",        means.get("MAP",        0), "avg precision over all relevant")
    _row(f"NDCG@{K}",       means.get("NDCG",      0), "graded rank-aware quality  <- gold standard")

    _sec("Diversity", "Intra-List Diversity  (Kaminskas & Bridge 2016)")
    _row("ILD",             means.get("ILD",       0), "0=identical  1=maximally diverse")

    core    = [means.get(m, 0) for m in ["Precision", "Recall", "MRR", "MAP", "NDCG"]]
    overall = float(np.mean(core))
    g, gc   = _grade(overall)
    print(f"\n  {'─'*56}")
    print(f"  {B}Overall  {gc}{overall:.4f}  [{g}]{R}  {_bar(overall)}")
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

    try:
        print(f"\n{D}Initializing engine for evaluation...{R}")
        # 1. Load data
        collection, df = run_full_pipeline(DATA_PATH, DB_PATH)
        
        # 2. Build engine
        build_engine(df)
        
        # 3. Run full evaluation
        run_evaluation()

    except Exception as e:
        print(f"\n{RE}Error during evaluation: {e}{R}")
        print(f"{Y}Usage:{R}")
        print("  from evaluate_engine import run_evaluation")
        print("  run_evaluation()                                        # default")
        print("  run_evaluation(queries=[(\"3 Idiots\", [\"hi\"], \"test\")]) # custom\n")