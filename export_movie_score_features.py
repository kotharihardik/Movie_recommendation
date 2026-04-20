"""
export_movie_score_features.py
------------------------------
Create a score-export CSV aligned with the main movie list row order.

Output columns:
- id
- title
- rating_norm
- vote_confidence
- fame_score
- popularity_score

Usage:
    python export_movie_score_features.py \
        --input data/movies.csv \
        --output data/movie_score_features.csv
"""

from __future__ import annotations

import argparse
import ast
import math
from collections import Counter

import numpy as np
import pandas as pd


def _safe_parse_list(val) -> list:
    """Parse stringified list safely; return [] on parse failure."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _compute_rating_norm(df: pd.DataFrame) -> np.ndarray:
    """
    Match engine logic:
      weighted_rating = (v/(v+m))*R + (m/(v+m))*C
      rating_norm = clip(weighted_rating, 0, 10)/10
    """
    vote_average = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0.0)
    vote_count = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0.0).clip(lower=0)

    c_val = float(vote_average.mean()) if len(vote_average) else 0.0
    m_val = float(vote_count.quantile(0.60)) if len(vote_count) else 0.0

    denom = vote_count + m_val
    weighted_rating = np.where(
        denom > 0,
        (vote_count / denom) * vote_average + (m_val / denom) * c_val,
        c_val,
    )
    return np.clip(weighted_rating, 0, 10) / 10.0


def _compute_vote_confidence(df: pd.DataFrame) -> np.ndarray:
    """Match semantic scorer logic: log-normalized vote_count confidence."""
    vc = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0.0).clip(lower=0)
    vc_max = float(vc.max()) if len(vc) else 0.0
    if vc_max <= 0:
        return np.zeros(len(df), dtype=float)
    return (np.log1p(vc) / np.log1p(vc_max + 1e-9)).to_numpy(dtype=float)


def _compute_fame_score(df: pd.DataFrame) -> np.ndarray:
    """
    Match engine fame heuristic:
            raw_fame = position-weighted cast appearance logs + 0.45 * director appearance log
            counts are built only from films with vote_count > 50
      fame_score = min-max(raw_fame)
    """
    work_df = df.copy()

    if "cast" not in work_df.columns:
        work_df["cast"] = [[] for _ in range(len(work_df))]
    if "director" not in work_df.columns:
        work_df["director"] = "Unknown"

    work_df["cast"] = work_df["cast"].apply(_safe_parse_list)
    work_df["cast"] = work_df["cast"].apply(lambda x: x[:5] if isinstance(x, list) else [])
    work_df["director"] = work_df["director"].fillna("Unknown").astype(str)

    vote_count = pd.to_numeric(work_df.get("vote_count", 0), errors="coerce").fillna(0).clip(lower=0)
    qualified_mask = vote_count > 50

    dir_counts: Counter = Counter()
    actor_counts: Counter = Counter()
    for (_, row), is_qualified in zip(work_df.iterrows(), qualified_mask):
        if not bool(is_qualified):
            continue
        director = str(row.get("director", "") or "")
        if director:
            dir_counts[director] += 1

        cast_list = row.get("cast", []) or []
        for actor in (cast_list or []):
            actor_counts[actor] += 1

    raw_fame = np.zeros(len(work_df), dtype=float)
    pos_weights = [1.0, 0.7, 0.5]
    for i, (_, row) in enumerate(work_df.iterrows()):
        cast_term = 0.0
        for pos, actor in enumerate((row.get("cast", []) or [])[:3]):
            w = pos_weights[pos] if pos < len(pos_weights) else 0.2
            cast_term += w * math.log1p(actor_counts.get(actor, 0))

        director = str(row.get("director", "") or "")
        dir_term = 0.45 * math.log1p(dir_counts.get(director, 0))
        raw_fame[i] = cast_term + dir_term

    rf_min = float(raw_fame.min()) if len(raw_fame) else 0.0
    rf_max = float(raw_fame.max()) if len(raw_fame) else 0.0
    if rf_max > rf_min:
        return (raw_fame - rf_min) / (rf_max - rf_min)
    return np.zeros(len(work_df), dtype=float)


def _compute_popularity_score(
    rating_norm: np.ndarray,
    vote_confidence: np.ndarray,
    fame_score: np.ndarray,
) -> np.ndarray:
    """Reliability-first popularity blend used across the app."""
    rating_arr = np.asarray(rating_norm, dtype=float)
    conf_arr = np.asarray(vote_confidence, dtype=float)
    fame_arr = np.asarray(fame_score, dtype=float)
    score = 0.55 * conf_arr + 0.30 * fame_arr + 0.15 * rating_arr
    return np.clip(score, 0.0, 1.0)


def export_scores(input_csv: str, output_csv: str) -> None:
    """Read input movie CSV, compute scores, and write output CSV in same row order."""
    df = pd.read_csv(input_csv, low_memory=False)

    # Keep row order exactly as input file.
    out = pd.DataFrame(index=df.index)
    out["id"] = df.get("id", pd.Series(range(len(df)))).astype(str)
    out["title"] = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)

    rating_norm = _compute_rating_norm(df)
    vote_confidence = _compute_vote_confidence(df)
    fame_score = _compute_fame_score(df)

    out["rating_norm"] = rating_norm
    out["vote_confidence"] = vote_confidence
    out["fame_score"] = fame_score
    out["popularity_score"] = _compute_popularity_score(rating_norm, vote_confidence, fame_score)

    out.to_csv(output_csv, index=False)
    print(f"Saved {len(out):,} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export movie score columns to CSV.")
    parser.add_argument(
        "--input",
        default="data/movies.csv",
        help="Path to main movie list CSV (default: data/movies.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/movie_score_features.csv",
        help="Output CSV path (default: data/movie_score_features.csv)",
    )
    args = parser.parse_args()

    export_scores(input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
