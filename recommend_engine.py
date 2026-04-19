"""
recommend_engine.py
-------------------
CineMatch India — Pure-Python recommendation engine.

Based exactly on the notebook approach (movie_recommendation_system_v3):
  1. TF-IDF content model  (weighted "soup" of keywords × genres × cast × director × overview)
  2. SBERT semantic model  (all-MiniLM-L6-v2 on natural-language descriptions)
  3. SVD + KNN collaborative model (latent keyword co-occurrence)
  4. Fame Score (star-power heuristic from cast/director appearance frequency)
  5. Hybrid fusion of all four signals

Query modes handled:
  • Movie name only  → anchor on that movie row, run hybrid
  • Free-text / mood / genre chips only → build synthetic SBERT query,
    no TF-IDF anchor (fall back to SBERT + CF + fame)
  • Combined → anchor on movie AND boost with free-text embedding

popularity / vote_count / vote_average columns are NOT used as ranking
signals; only the fame_score derived from cast/director frequency matters.
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# ── Data class for a single recommended movie ─────────────────────────────────

@dataclass
class RecommendedMovie:
    movie_id:       str
    title:          str
    original_title: str
    language:       str
    year:           int
    runtime:        int
    vote_average:   float
    vote_count:     int
    genres:         list
    director:       str
    cast:           list
    poster_path:    str
    tagline:        str
    overview:       str
    budget:         int
    revenue:        int
    weighted_score: float        # normalised [0,1] shown as match %
    fame_score:     float        # popularity signal (0-1, higher = more famous)
    justification:  str = ""


# ── Module-level state (populated by build_engine once) ──────────────────────

_engine_ready:  bool = False
_df:            Optional[pd.DataFrame] = None

_tfidf_matrix   = None
_tfidf:         Optional[TfidfVectorizer] = None
_title_to_idx:  Optional[pd.Series] = None

_sbert          = None
_sbert_vecs     = None

_knn            = None
_movie_vecs     = None

_fame_scores:   Optional[np.ndarray] = None   # aligned with _df index
_vote_priority_scores: Optional[np.ndarray] = None   # aligned with _df index


def _fmt_list(values, max_items: int = 8) -> str:
    """Format list-like values for concise terminal debug output."""
    if not isinstance(values, list) or not values:
        return "-"
    shown = values[:max_items]
    extra = "" if len(values) <= max_items else f" ... (+{len(values) - max_items} more)"
    return ", ".join(str(v) for v in shown) + extra


def _debug_movie_meta(label: str, row: pd.Series) -> None:
    """Print essential movie metadata for debugging recommendation decisions."""
    print(f"\\n[DEBUG] {label}")
    print(f"  title       : {row.get('title', '')}")
    print(f"  year/lang   : {row.get('release_year', 0)} / {row.get('language', '')}")
    print(f"  vote        : avg={float(row.get('vote_average', 0.0)):.2f}, count={int(row.get('vote_count', 0))}")
    print(f"  genres      : {_fmt_list(row.get('genres', []), max_items=10)}")
    print(f"  keywords    : {_fmt_list(row.get('keywords', []), max_items=12)}")
    print(f"  cast        : {_fmt_list(row.get('cast', []), max_items=8)}")


def _safe_float(v, default: float = 0.0) -> float:
    """Convert values to float while mapping NaN/inf/None to default."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


# ── Token helpers (identical to notebook) ────────────────────────────────────

def _clean_token(s: str) -> str:
    """Strip punctuation, lowercase — 'Shah Rukh Khan' → 'shahrukhkhan'."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()


_STOP = {
    'a','an','the','and','or','but','is','are','was','were','be','been',
    'being','have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','to','of','in','on','at','by',
    'for','with','about','as','into','through','his','her','their','its',
    'he','she','they','we','it','this','that','these','those','who','which',
    'when','where','how','what','not','no','nor','so','yet','both','either',
    'from','up','out','if','then','than','too','very','just','also'
}

def _tokenise_text(text: str) -> str:
    tokens = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return ' '.join(t for t in tokens if t not in _STOP)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity with safe empty-set handling."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _is_clean_title(x: str) -> bool:
    """Filter obvious noisy titles (empty, numeric-only, symbol-only)."""
    t = str(x).strip()
    if len(t) < 2:
        return False
    if re.fullmatch(r'[0-9]+', t):
        return False
    if re.fullmatch(r'[^a-zA-Z0-9]+', t):
        return False
    return True


# ── Feature builders (identical to notebook) ─────────────────────────────────

def _make_soup(row: pd.Series) -> str:
    """
    Weighted feature soup for TF-IDF  (from notebook):
      keywords ×3 · genres ×3 · cast ×2 · director ×1 · overview ×2
    """
    kw_tok   = ' '.join(_clean_token(k) for k in (row['keywords'] or []))
    kw_w     = ' '.join([kw_tok] * 3)

    g_tok    = ' '.join(_clean_token(g) for g in (row['genres'] or []))
    g_w      = ' '.join([g_tok] * 3)

    c_tok    = ' '.join(_clean_token(a) for a in (row['cast'] or []))
    c_w      = ' '.join([c_tok] * 2)

    d_tok    = _clean_token(row.get('director', '') or '')

    ov_tok   = _tokenise_text(str(row.get('overview', '') or ''))
    ov_w     = f'{ov_tok} {ov_tok}'

    return f'{kw_w} {g_w} {c_w} {d_tok} {ov_w}'.strip()


def _make_description(row: pd.Series) -> str:
    """Natural-language description for SBERT  (from notebook)."""
    genres   = row.get('genres', []) or []
    keywords = row.get('keywords', []) or []
    cast     = row.get('cast', []) or []
    director = row.get('director', 'Unknown') or 'Unknown'
    overview = str(row.get('overview', '') or '')

    genre_str = ', '.join(genres) if genres else 'various genres'
    kw_str    = ', '.join(keywords) if keywords else ''
    cast_str  = ', '.join(cast) if cast else 'an ensemble cast'
    dir_str   = director if director != 'Unknown' else 'an unknown director'
    year      = str(row.get('release_year', '') or '')

    desc = (
        f"A {genre_str} film released in {year}. "
        f"This movie strongly belongs to the genres: {genre_str}. "
        f"Directed by {dir_str} and starring {cast_str}. "
    )
    if overview:
        desc += overview + ' '
    if kw_str:
        desc += f"Key themes include: {kw_str}."
    return desc.strip()


# ── Vote-count priority (year-normalized) ───────────────────────────────────

def _compute_fame_scores(df: pd.DataFrame) -> np.ndarray:
        """
        Star-power fame heuristic (does NOT use popularity column).

        Logic:
            • For each actor / director count how many movies they appear in across
                the whole dataset → appearance frequency (log-normalised)
            • A movie's raw fame = mean(top-3 cast freq) * 0.55 + director_freq * 0.45
            • Final score is MinMaxScaler → [0, 1]
        """
        dir_counts = df['director'].value_counts().to_dict()

        from collections import Counter
        actor_counts: Counter = Counter()
        for cast_list in df['cast']:
                for actor in (cast_list or []):
                        actor_counts[actor] += 1

        raw_fame = np.zeros(len(df), dtype=float)

        for i, (_, row) in enumerate(df.iterrows()):
                dir_score = math.log1p(dir_counts.get(row.get('director', ''), 0))
                cast_list = [a for a in (row.get('cast', []) or [])][:5]
                cast_freqs = sorted(
                        [math.log1p(actor_counts.get(a, 0)) for a in cast_list],
                        reverse=True
                )
                cast_score = np.mean(cast_freqs[:3]) if cast_freqs else 0.0
                raw_fame[i] = 0.55 * cast_score + 0.45 * dir_score

        scaler = MinMaxScaler()
        return scaler.fit_transform(raw_fame.reshape(-1, 1)).flatten()


# ── Vote-count priority (year-normalized) ───────────────────────────────────

def _compute_year_normalized_vote_priority(df: pd.DataFrame) -> np.ndarray:
    """
    Build a vote-count priority score in [0, 1] with year-wise normalisation.

    Why year-wise normalisation:
      older films naturally accumulate more votes than recent releases.
      To avoid unfairly penalising newer movies, we rank vote_count inside each
      release year and blend it with a small global vote_count signal.
    """
    vc = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).clip(lower=0)

    # Year-wise percentile rank in [0,1]. Single-item years get 1.0.
    year_rank = (
        df.assign(_vc=vc)
        .groupby('release_year')['_vc']
        .rank(method='average', pct=True)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # Global vote_count signal (log scaled then min-max), prevents overfitting
    # to tiny year buckets.
    global_log_vc = np.log1p(vc.to_numpy(dtype=float))
    scaler = MinMaxScaler()
    global_norm = scaler.fit_transform(global_log_vc.reshape(-1, 1)).flatten()

    # Blend: mostly year-wise fairness + some global confidence.
    return 0.75 * year_rank + 0.25 * global_norm


# ── Engine builder ────────────────────────────────────────────────────────────

def build_engine(df: pd.DataFrame) -> None:
    """
    Build all recommendation models from the cleaned dataframe.
    Called once at startup (or when the engine hasn't been built yet).
    """
    global _engine_ready, _df
    global _tfidf_matrix, _tfidf, _title_to_idx
    global _sbert, _sbert_vecs
    global _knn, _movie_vecs
    global _fame_scores
    global _vote_priority_scores

    if _engine_ready:
        return

    print("🎬 Building recommendation engine…")
    t0 = time.time()

    # ── Store df ─────────────────────────────────────────────────────────────
    _df = df.reset_index(drop=True).copy()

    # ── Derived columns needed by notebook formulas ───────────────────────────
    C = _df['vote_average'].mean()
    m = _df['vote_count'].quantile(0.60)
    _df['weighted_rating'] = (
        (_df['vote_count'] / (_df['vote_count'] + m)) * _df['vote_average'] +
        (m / (_df['vote_count'] + m)) * C
    )

    # ── Soup + description ────────────────────────────────────────────────────
    print("  [1/4] Building TF-IDF soup…")
    _df['soup']        = _df.apply(_make_soup, axis=1)
    _df['description'] = _df.apply(_make_description, axis=1)

    # ── TF-IDF model ─────────────────────────────────────────────────────────
    _tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )
    _tfidf_matrix = _tfidf.fit_transform(_df['soup'])
    _title_to_idx  = pd.Series(_df.index, index=_df['title'].str.lower().str.strip())
    print(f"     TF-IDF shape: {_tfidf_matrix.shape}")

    # ── SBERT ─────────────────────────────────────────────────────────────────
    print("  [2/4] Encoding SBERT embeddings…")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        _sbert_vecs = _sbert.encode(
            _df['description'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"     SBERT shape: {_sbert_vecs.shape}")
    except Exception as e:
        print(f"     ⚠️  SBERT unavailable ({e}); semantic scoring disabled.")
        _sbert = None
        _sbert_vecs = None

    # ── SVD + KNN collaborative model ─────────────────────────────────────────
    print("  [3/4] Building SVD+KNN collaborative model…")
    try:
        cv = CountVectorizer(
            analyzer='word',
            ngram_range=(1, 1),
            min_df=2,
            max_features=15_000,
        )
        kw_text = _df.apply(
            lambda r: ' '.join(
                [_clean_token(k) for k in (r['keywords'] or [])] +
                [_clean_token(g) for g in (r['genres'] or [])] +
                [_clean_token(a) for a in (r['cast'] or [])]
            ),
            axis=1,
        )
        kw_matrix = cv.fit_transform(kw_text)

        n_comp  = min(300, kw_matrix.shape[1] - 1)
        svd     = TruncatedSVD(n_components=n_comp, random_state=42)
        _movie_vecs = svd.fit_transform(kw_matrix)

        _knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=100)
        _knn.fit(_movie_vecs)
        print(f"     SVD shape: {_movie_vecs.shape}, KNN ready")
    except Exception as e:
        print(f"     ⚠️  SVD/KNN unavailable ({e})")
        _knn = None
        _movie_vecs = None

    # ── Fame + vote-count priority scores ────────────────────────────────────
    print("  [4/4] Computing fame and vote-count priority scores…")
    _fame_scores = _compute_fame_scores(_df)
    _vote_priority_scores = _compute_year_normalized_vote_priority(_df)
    print(
        f"     Fame range: [{_fame_scores.min():.3f}, {_fame_scores.max():.3f}] | "
        f"Vote-priority range: "
        f"[{_vote_priority_scores.min():.3f}, {_vote_priority_scores.max():.3f}]"
    )

    _engine_ready = True
    print(f"✅ Engine ready in {time.time()-t0:.1f}s")


# ── Movie lookup ──────────────────────────────────────────────────────────────

def _find_movie_idx(query: str, language: str = None) -> Optional[int]:
    """Fuzzy lookup: exact → startswith → contains.  Returns None if not found."""
    if _title_to_idx is None or _df is None:
        return None
    q = query.lower().strip()

    def pick(indices_like):
        idxs = list(indices_like.values) if isinstance(indices_like, pd.Series) else [int(indices_like)]
        if language:
            for i in idxs:
                if _df.loc[i, 'language'] == language:
                    return int(i)
        return int(idxs[0])

    if q in _title_to_idx.index:
        return pick(_title_to_idx[q])
    candidates = [k for k in _title_to_idx.index if k.startswith(q)]
    if candidates:
        return pick(_title_to_idx[candidates[0]])
    candidates = [k for k in _title_to_idx.index if q in k]
    if candidates:
        return pick(_title_to_idx[candidates[0]])
    return None


# ── Sub-scorers ───────────────────────────────────────────────────────────────

def _tfidf_scores(anchor_idx: int) -> np.ndarray:
    """TF-IDF cosine similarity against the anchor movie."""
    if _tfidf_matrix is None:
        return np.zeros(len(_df))
    sims = linear_kernel(_tfidf_matrix[anchor_idx], _tfidf_matrix).flatten()
    sims[anchor_idx] = 0.0
    return sims


def _sbert_scores_from_idx(anchor_idx: int) -> np.ndarray:
    """SBERT cosine similarity against the anchor movie (embeddings are L2-normalised)."""
    if _sbert_vecs is None:
        return np.zeros(len(_df))
    sims = (_sbert_vecs @ _sbert_vecs[anchor_idx]).flatten()
    sims[anchor_idx] = 0.0
    return sims


def _sbert_scores_from_text(query_text: str) -> np.ndarray:
    """Encode arbitrary query text with SBERT and compute cosine similarity."""
    if _sbert_vecs is None or _sbert is None:
        return np.zeros(len(_df))
    q_vec = _sbert.encode([query_text], normalize_embeddings=True)[0]
    return (_sbert_vecs @ q_vec).flatten()


def _knn_scores(anchor_idx: int) -> np.ndarray:
    """SVD+KNN collaborative scores for the anchor movie."""
    out = np.zeros(len(_df))
    if _knn is None or _movie_vecs is None:
        return out
    dists, nn_idx = _knn.kneighbors(_movie_vecs[anchor_idx].reshape(1, -1), n_neighbors=100)
    for dist, idx in zip(dists.flatten(), nn_idx.flatten()):
        if idx != anchor_idx and idx < len(out):
            out[idx] = max(out[idx], 1.0 - dist)
    return out


# ── Chip / free-text query builder ───────────────────────────────────────────

def _build_query_text(free_text: str, selected_chips: list) -> str:
    """Merge free-text description + genre/mood chips into one SBERT query string."""
    parts = []
    if selected_chips:
        genres  = [c for c in selected_chips if c in {
            "Action","Romance","Thriller","Drama","Comedy",
            "Horror","Family","Historical","Crime","Sci-Fi"
        }]
        moods   = [c for c in selected_chips if c not in genres]
        if genres:
            parts.append(f"A {', '.join(genres)} film.")
        if moods:
            parts.append(f"The mood is {', '.join(moods).lower()}.")
    if free_text:
        parts.append(free_text.strip())
    return ' '.join(parts)


# ── Year-decade filter ────────────────────────────────────────────────────────

def _decade_mask(df: pd.DataFrame, decade_filter: list) -> np.ndarray:
    """Return boolean mask aligned with df for selected decades."""
    if not decade_filter:
        return np.ones(len(df), dtype=bool)
    masks = []
    for d in decade_filter:
        if d == "2020s":
            masks.append((df['release_year'] >= 2020).values)
        elif d == "2010s":
            masks.append(((df['release_year'] >= 2010) & (df['release_year'] < 2020)).values)
        elif d == "2000s":
            masks.append(((df['release_year'] >= 2000) & (df['release_year'] < 2010)).values)
        elif d == "1990s":
            masks.append(((df['release_year'] >= 1990) & (df['release_year'] < 2000)).values)
        elif "Classic" in d:
            masks.append((df['release_year'] < 1990).values)
    if not masks:
        return np.ones(len(df), dtype=bool)
    return np.any(np.stack(masks, axis=0), axis=0)


# ── Jaccard genre/keyword overlap helpers ─────────────────────────────────────

def _genre_overlap_scores(df: pd.DataFrame, query_genres: set) -> np.ndarray:
    """Jaccard similarity between query_genres and each movie's genres."""
    if not query_genres:
        return np.zeros(len(df))
    return np.array([
        _jaccard(query_genres, set(g) if isinstance(g, list) else set())
        for g in df['genres']
    ])


def _normalise_title_for_franchise(title: str) -> str:
    """Normalize titles to a franchise/base form (e.g., 'Dabangg 3' -> 'dabangg')."""
    t = re.sub(r'[^a-z0-9 ]+', ' ', str(title).lower())
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return ''

    # Remove trailing sequel markers like numbers / roman numerals.
    t = re.sub(r'\b(?:part|chapter|episode)\b\s*[a-z0-9ivx]*$', '', t).strip()
    t = re.sub(r'\b(?:[0-9]+|[ivx]+)\b$', '', t).strip()
    return t


def _franchise_boost_scores(df: pd.DataFrame, anchor_title: str) -> np.ndarray:
    """Return per-row [0,1] title-franchise similarity to anchor title."""
    anchor_base = _normalise_title_for_franchise(anchor_title)
    if not anchor_base:
        return np.zeros(len(df), dtype=float)

    anchor_tokens = {t for t in anchor_base.split() if len(t) >= 3}
    if not anchor_tokens:
        anchor_tokens = set(anchor_base.split())

    scores = np.zeros(len(df), dtype=float)
    for i, t in enumerate(df['title'].fillna('').astype(str).tolist()):
        cand_base = _normalise_title_for_franchise(t)
        if not cand_base:
            continue
        if cand_base == anchor_base:
            scores[i] = 1.0
            continue
        # Prefix containment is a strong franchise cue, e.g. "border" vs "border 2".
        if cand_base.startswith(anchor_base) or anchor_base.startswith(cand_base):
            scores[i] = max(scores[i], 0.95)
            continue
        cand_tokens = {x for x in cand_base.split() if len(x) >= 3}
        if not cand_tokens:
            cand_tokens = set(cand_base.split())
        scores[i] = _jaccard(anchor_tokens, cand_tokens)
    return scores


def _semantic_recommend_from_anchor(
    anchor_idx: int,
    language_codes: list,
    decade_filter: list,
    top_n: int,
    min_vote_avg: float = 5.0,
    year_window: int = 12,
    min_genre_overlap: int = 2,
    min_vote_count: int = 20,
) -> list[RecommendedMovie]:
    """
    Semantic recommender for title-only queries.
    Uses SBERT similarity with overlap/time/quality guardrails.
    """
    if _sbert_vecs is None or _df is None:
        return []

    df = _df
    q_row = df.loc[anchor_idx]
    anchor_title_exact = str(q_row.get('title', '')).strip().lower()

    # SBERT cosine (embeddings are already L2-normalized)
    sbert_sim = (_sbert_vecs @ _sbert_vecs[anchor_idx]).flatten()
    sbert_sim[anchor_idx] = 0.0

    q_genres = set(q_row['genres']) if isinstance(q_row.get('genres'), list) else set()
    q_keywords = set(q_row['keywords']) if isinstance(q_row.get('keywords'), list) else set()
    q_cast = set((q_row.get('cast') or [])[:5]) if isinstance(q_row.get('cast'), list) else set()
    q_year = int(q_row.get('release_year', 2000))

    print("\n[DEBUG][SEMANTIC] =============================================")
    _debug_movie_meta("Input movie", q_row)

    candidate_df = df.copy()
    candidate_df['sbert_raw'] = sbert_sim
    franchise_scores = _franchise_boost_scores(df, str(q_row.get('title', '')))
    candidate_df['franchise_boost'] = pd.Series(franchise_scores, index=df.index)
    candidate_df = candidate_df[candidate_df['title'].apply(_is_clean_title)]
    # Never recommend the exact same movie title in title-search mode.
    candidate_df = candidate_df[
        candidate_df['title'].fillna('').astype(str).str.strip().str.lower() != anchor_title_exact
    ]

    candidate_df['genre_overlap'] = candidate_df['genres'].apply(
        lambda g: len(q_genres & set(g)) if isinstance(g, list) else 0
    )
    candidate_df['genre_jaccard'] = candidate_df['genres'].apply(
        lambda g: _jaccard(q_genres, set(g)) if isinstance(g, list) else 0.0
    )

    # First hard filter for movie-title mode: require strong genre alignment.
    # If query has 2+ genres, require at least 2 overlap; otherwise require 1.
    required_genre_overlap = min(min_genre_overlap, len(q_genres)) if q_genres else 0
    if q_genres:
        candidate_df = candidate_df[
            (candidate_df['genre_overlap'] >= required_genre_overlap)
        ]

    if 'keywords' in candidate_df.columns:
        candidate_df['keyword_jaccard'] = candidate_df['keywords'].apply(
            lambda k: _jaccard(q_keywords, set(k)) if isinstance(k, list) else 0.0
        )
    else:
        candidate_df['keyword_jaccard'] = 0.0

    candidate_df['cast_jaccard'] = candidate_df['cast'].apply(
        lambda c: _jaccard(q_cast, set(c[:5])) if isinstance(c, list) else 0.0
    )

    year_diff = np.abs(candidate_df['release_year'].astype(float) - float(q_year))
    sigma = float(max(year_window, 1))
    candidate_df['temporal_soft'] = np.exp(-(year_diff ** 2) / (2.0 * (sigma ** 2)))

    vc = pd.to_numeric(candidate_df['vote_count'], errors='coerce').fillna(0).clip(lower=0)
    vc_max = float(vc.max())
    candidate_df['vote_confidence'] = (
        np.log1p(vc) / np.log1p(vc_max + 1e-9)
        if vc_max > 0 else 0.0
    )
    vote_priority_all = _vote_priority_scores if _vote_priority_scores is not None else np.zeros(len(df))
    candidate_df['vote_priority_year'] = pd.Series(vote_priority_all, index=df.index).reindex(candidate_df.index).fillna(0.0)
    candidate_df['rating_norm'] = (
        pd.to_numeric(candidate_df['weighted_rating'], errors='coerce')
        .fillna(0)
        .clip(lower=0, upper=10) / 10.0
    )

    candidate_df['semantic_score'] = (
        0.36 * candidate_df['sbert_raw'] +
        0.16 * candidate_df['genre_jaccard'] +
        0.06 * candidate_df['cast_jaccard'] +
        0.03 * candidate_df['keyword_jaccard'] +
        0.05 * candidate_df['temporal_soft'] +
        0.14 * candidate_df['vote_confidence'] +
        0.14 * candidate_df['vote_priority_year'] +
        0.01 * candidate_df['rating_norm'] +
        0.05 * candidate_df['franchise_boost']
    )

    if anchor_idx in candidate_df.index:
        candidate_df.loc[anchor_idx, 'semantic_score'] = 0.0

    if language_codes:
        candidate_df = candidate_df[candidate_df['language'].isin(language_codes)]

    dec_mask = _decade_mask(candidate_df, decade_filter)
    candidate_df = candidate_df[dec_mask]

    candidate_df = candidate_df[
        (pd.to_numeric(candidate_df['vote_average'], errors='coerce').fillna(0) >= min_vote_avg) |
        (candidate_df['franchise_boost'] >= 0.75)
    ]

    # Never recommend movies with zero votes.
    candidate_df = candidate_df[
        pd.to_numeric(candidate_df['vote_count'], errors='coerce').fillna(0) > 0
    ]

    # Hard vote-count floor for movie-title mode.
    candidate_df = candidate_df[
        (pd.to_numeric(candidate_df['vote_count'], errors='coerce').fillna(0) > min_vote_count) |
        (candidate_df['franchise_boost'] >= 0.75)
    ]

    # Strong same-name/franchise matches should appear first, but we still fill
    # remaining slots with other good recommendations.
    strong_name_df = candidate_df[candidate_df['franchise_boost'] >= 0.75]

    in_window = candidate_df[
        np.abs(pd.to_numeric(candidate_df['release_year'], errors='coerce').fillna(0) - q_year) <= year_window
    ]
    if len(in_window) >= top_n:
        candidate_df = in_window

    if len(candidate_df) < top_n:
        relaxed_df = df.copy()
        relaxed_df = relaxed_df[relaxed_df['title'].apply(_is_clean_title)]
        relaxed_df = relaxed_df[
            relaxed_df['title'].fillna('').astype(str).str.strip().str.lower() != anchor_title_exact
        ]
        # Align SBERT scores to the filtered frame index to avoid shape mismatch.
        relaxed_df['sbert_raw'] = pd.Series(sbert_sim, index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['franchise_boost'] = pd.Series(franchise_scores, index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['genre_jaccard'] = relaxed_df['genres'].apply(
            lambda g: _jaccard(q_genres, set(g)) if isinstance(g, list) else 0.0
        )
        relaxed_df['cast_jaccard'] = relaxed_df['cast'].apply(
            lambda c: _jaccard(q_cast, set(c[:5])) if isinstance(c, list) else 0.0
        )
        relaxed_df['keyword_jaccard'] = relaxed_df['keywords'].apply(
            lambda k: _jaccard(q_keywords, set(k)) if isinstance(k, list) else 0.0
        ) if 'keywords' in relaxed_df.columns else 0.0
        relaxed_year_diff = np.abs(
            pd.to_numeric(relaxed_df['release_year'], errors='coerce').fillna(0) - q_year
        )
        relaxed_df['temporal_soft'] = np.exp(-(relaxed_year_diff ** 2) / (2.0 * (float(max(year_window, 1)) ** 2)))
        relaxed_vc = pd.to_numeric(relaxed_df['vote_count'], errors='coerce').fillna(0).clip(lower=0)
        relaxed_vc_max = float(relaxed_vc.max())
        relaxed_df['vote_confidence'] = (
            np.log1p(relaxed_vc) / np.log1p(relaxed_vc_max + 1e-9)
            if relaxed_vc_max > 0 else 0.0
        )
        relaxed_df['vote_priority_year'] = pd.Series(vote_priority_all, index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['rating_norm'] = (
            pd.to_numeric(relaxed_df['weighted_rating'], errors='coerce')
            .fillna(0)
            .clip(lower=0, upper=10) / 10.0
        )
        relaxed_df['semantic_score'] = (
            0.36 * relaxed_df['sbert_raw'] +
            0.16 * relaxed_df['genre_jaccard'] +
            0.06 * relaxed_df['cast_jaccard'] +
            0.03 * relaxed_df['keyword_jaccard'] +
            0.05 * relaxed_df['temporal_soft'] +
            0.14 * relaxed_df['vote_confidence'] +
            0.14 * relaxed_df['vote_priority_year'] +
            0.01 * relaxed_df['rating_norm'] +
            0.05 * relaxed_df['franchise_boost']
        )
        if anchor_idx in relaxed_df.index:
            relaxed_df.loc[anchor_idx, 'semantic_score'] = 0.0
        if language_codes:
            relaxed_df = relaxed_df[relaxed_df['language'].isin(language_codes)]
        relaxed_dec_mask = _decade_mask(relaxed_df, decade_filter)
        relaxed_df = relaxed_df[relaxed_dec_mask]
        if q_genres:
            # Softer fill-stage filter: allow broader related titles to complete
            # the requested result count.
            relaxed_df = relaxed_df[
                relaxed_df['genres'].apply(
                    lambda g: len(q_genres & set(g)) if isinstance(g, list) else 0
                ) >= required_genre_overlap
            ]
        relaxed_df = relaxed_df[
            pd.to_numeric(relaxed_df['vote_count'], errors='coerce').fillna(0) > 0
        ]
        relaxed_df = relaxed_df[
            (pd.to_numeric(relaxed_df['vote_count'], errors='coerce').fillna(0) > min_vote_count) |
            (relaxed_df['franchise_boost'] >= 0.75)
        ]
        relaxed_df = relaxed_df[
            (pd.to_numeric(relaxed_df['vote_average'], errors='coerce').fillna(0) >= min_vote_avg) |
            (relaxed_df['franchise_boost'] >= 0.75)
        ]
        candidate_df = pd.concat([candidate_df, relaxed_df], axis=0)
        candidate_df = candidate_df[~candidate_df.index.duplicated(keep='first')]

    # Ensure numeric dtypes for robust ranking with nlargest.
    candidate_df['semantic_score'] = pd.to_numeric(
        candidate_df.get('semantic_score', 0.0), errors='coerce'
    ).fillna(0.0)
    candidate_df['franchise_boost'] = pd.to_numeric(
        candidate_df.get('franchise_boost', 0.0), errors='coerce'
    ).fillna(0.0)
    for col in [
        'sbert_raw',
        'genre_jaccard',
        'cast_jaccard',
        'keyword_jaccard',
        'temporal_soft',
        'vote_confidence',
        'vote_priority_year',
        'rating_norm',
    ]:
        candidate_df[col] = pd.to_numeric(candidate_df.get(col, 0.0), errors='coerce').fillna(0.0)

    # Final ranking: reserve slots for strong franchise matches first, then fill.
    # This keeps sequel/series continuity without hardcoding any specific movie.
    franchise_quota = min(3, top_n)
    franchise_df = candidate_df[candidate_df['franchise_boost'] >= 0.90].sort_values(
        by=['semantic_score', 'vote_priority_year', 'vote_count'],
        ascending=[False, False, False],
    ).head(franchise_quota)
    remaining_df = candidate_df.drop(index=franchise_df.index, errors='ignore').sort_values(
        by=['semantic_score', 'vote_priority_year', 'vote_count'],
        ascending=[False, False, False],
    )
    top = pd.concat([franchise_df, remaining_df], axis=0).head(top_n)
    fame_norm = _fame_scores if _fame_scores is not None else np.zeros(len(df))

    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        cand_genres = set(row.get('genres', []) if isinstance(row.get('genres', []), list) else [])
        cand_keywords = set(row.get('keywords', []) if isinstance(row.get('keywords', []), list) else [])
        cand_cast = set((row.get('cast', []) or [])[:5]) if isinstance(row.get('cast', []), list) else set()

        shared_genres = sorted(list(q_genres & cand_genres))
        shared_keywords = sorted(list(q_keywords & cand_keywords))
        shared_cast = sorted(list(q_cast & cand_cast))

        sbert_raw = _safe_float(row.get('sbert_raw', 0.0))
        genre_j = _safe_float(row.get('genre_jaccard', 0.0))
        cast_j = _safe_float(row.get('cast_jaccard', 0.0))
        keyword_j = _safe_float(row.get('keyword_jaccard', 0.0))
        temporal = _safe_float(row.get('temporal_soft', 0.0))
        vote_conf = _safe_float(row.get('vote_confidence', 0.0))
        vote_priority_year = _safe_float(row.get('vote_priority_year', 0.0))
        rating_norm = _safe_float(row.get('rating_norm', 0.0))
        franchise = _safe_float(row.get('franchise_boost', 0.0))
        semantic = _safe_float(row.get('semantic_score', 0.0))

        comp = {
            'sbert(0.36)': 0.36 * sbert_raw,
            'genre_jaccard(0.16)': 0.16 * genre_j,
            'cast_jaccard(0.06)': 0.06 * cast_j,
            'keyword_jaccard(0.03)': 0.03 * keyword_j,
            'temporal_soft(0.05)': 0.05 * temporal,
            'vote_confidence(0.14)': 0.14 * vote_conf,
            'vote_priority_year(0.14)': 0.14 * vote_priority_year,
            'rating_norm(0.01)': 0.01 * rating_norm,
            'franchise_boost(0.05)': 0.05 * franchise,
        }
        top_reasons = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)[:3]

        print(f"\n[DEBUG][SEMANTIC][RANK {rank}] {row.get('title', '')}")
        print(
            f"  final_semantic_score={semantic:.4f} | "
            f"vote_count={int(row.get('vote_count', 0))} | "
            f"vote_avg={float(row.get('vote_average', 0.0)):.2f}"
        )
        print(
            f"  components raw: sbert={sbert_raw:.4f}, genre_jaccard={genre_j:.4f}, "
            f"cast_jaccard={cast_j:.4f}, keyword_jaccard={keyword_j:.4f}, "
            f"temporal_soft={temporal:.4f}, vote_conf={vote_conf:.4f}, vote_priority_year={vote_priority_year:.4f}, "
            f"rating_norm={rating_norm:.4f}, franchise_boost={franchise:.4f}"
        )
        print(
            f"  weighted contribution: "
            + ", ".join(f"{k}={v:.4f}" for k, v in comp.items())
        )
        print(
            "  why recommended: "
            + ", ".join(f"{k}={v:.4f}" for k, v in top_reasons)
        )
        print(f"  genre         : {_fmt_list(row.get('genres', []), max_items=10)}")
        print(f"  keyword       : {_fmt_list(row.get('keywords', []), max_items=12)}")
        print(f"  cast          : {_fmt_list(row.get('cast', []), max_items=8)}")
        print(f"  shared_genres : {_fmt_list(shared_genres, max_items=10)}")
        print(f"  shared_keywords: {_fmt_list(shared_keywords, max_items=12)}")
        print(f"  shared_cast   : {_fmt_list(shared_cast, max_items=8)}")

    results: list[RecommendedMovie] = []
    for idx, row in top.iterrows():
        score = float(max(0.0, min(1.0, row.get('semantic_score', 0.0))))
        fm = float(fame_norm[idx]) if idx < len(fame_norm) else 0.0

        results.append(RecommendedMovie(
            movie_id       = str(row.get('id', idx)),
            title          = str(row.get('title', '')),
            original_title = str(row.get('original_title', '')),
            language       = str(row.get('language', '')),
            year           = int(row.get('release_year', 0)),
            runtime        = int(row.get('runtime', 0)),
            vote_average   = float(row.get('vote_average', 0)),
            vote_count     = int(row.get('vote_count', 0)),
            genres         = list(row.get('genres', []) or []),
            director       = str(row.get('director', '')),
            cast           = list(row.get('cast', []) or []),
            poster_path    = str(row.get('poster_path', '')),
            tagline        = str(row.get('tagline', '')),
            overview       = str(row.get('overview', '')),
            budget         = int(row.get('budget', 0)),
            revenue        = int(row.get('revenue', 0)),
            weighted_score = score,
            fame_score     = fm,
        ))

    return results


# ── Hybrid recommender (core) ─────────────────────────────────────────────────

def _hybrid_recommend(
    anchor_idx:      Optional[int],
    query_text:      str,
    query_genres:    set,
    language_codes:  list,
    decade_filter:   list,
    top_n:           int,
    min_vote_avg:    float = 5.0,    # do not recommend movies rated below this
    genre_only_mode: bool = False,
    diversify:       bool = False,
) -> list[RecommendedMovie]:
    """
    Core hybrid engine.

    Score = w1*tfidf + w2*sbert + w3*cf + fame_boost
    Weights depend on which signals are available.
    """
    df = _df
    n  = len(df)
    genre_bonus = np.zeros(n, dtype=float)

    # ── Base similarity scores ────────────────────────────────────────────────
    if anchor_idx is not None:
        q_title = str(df.loc[anchor_idx].get('title', ''))
        franchise_boost = _franchise_boost_scores(df, q_title)

        s_tfidf = _tfidf_scores(anchor_idx)
        s_sbert = (
            _sbert_scores_from_idx(anchor_idx)
            if _sbert_vecs is not None else np.zeros(n)
        )
        s_cf    = _knn_scores(anchor_idx)

        # If there's also free text, blend it in (20% weight on top)
        if query_text:
            s_text_extra = _sbert_scores_from_text(query_text)
            s_sbert = 0.80 * s_sbert + 0.20 * s_text_extra

        # Weights: TF-IDF 35%, SBERT 45%, CF 20%
        # (replicates notebook: content 0.8 + semantic 0.5 + cf blend)
        total_sim = 0.35 * s_tfidf + 0.60 * s_sbert + 0.05 * s_cf

        # Anchor movie's genre set for structural overlap guardrail
        q_row      = df.loc[anchor_idx]
        q_genres_a = set(q_row['genres']) if isinstance(q_row.get('genres'), list) else set()
        q_keywords_a = set(q_row['keywords']) if isinstance(q_row.get('keywords'), list) else set()
        q_cast_a = set((q_row.get('cast') or [])[:5]) if isinstance(q_row.get('cast'), list) else set()
        q_year     = int(q_row.get('release_year', 2000))

        print("\n[DEBUG][HYBRID] ===============================================")
        _debug_movie_meta("Input movie", q_row)

    else:
        # No anchor movie → pure text/chip query
        franchise_boost = np.zeros(n)
        s_sbert     = _sbert_scores_from_text(query_text) if query_text else np.zeros(n)
        s_cf        = np.zeros(n)
        s_tfidf     = np.zeros(n)
        total_sim   = s_sbert
        q_genres_a  = set()
        q_keywords_a = set()
        q_cast_a = set()
        q_year      = None

        print("\n[DEBUG][HYBRID] ===============================================")
        print("[DEBUG] Input query (no anchor movie)")
        print(f"  query_text   : {query_text if query_text else '-'}")
        print(f"  query_genres : {_fmt_list(sorted(list(query_genres)), max_items=10)}")

    # ── Genre overlap bonus (structural guardrail, from notebook) ─────────────
    effective_genres = query_genres | q_genres_a
    if effective_genres:
        genre_bonus = _genre_overlap_scores(df, effective_genres)
        total_sim  = 0.82 * total_sim + 0.18 * genre_bonus

    # ── Final ranking integration ─────────────────────────────────────────────
    # Keep fame boost, remove popularity-column dependency, and add year-wise
    # vote-count priority.
    sim_norm  = total_sim / (total_sim.max() + 1e-9)
    fame_norm = _fame_scores if _fame_scores is not None else np.zeros(n)
    vote_priority = (
        _vote_priority_scores
        if _vote_priority_scores is not None else np.zeros(n)
    )

    rating_norm = (
        pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).clip(lower=0, upper=10).values / 10.0
    )

    if genre_only_mode:
        # For pure genre-tag queries, prioritize widely-voted and highly-rated movies.
        final = 0.55 * vote_priority + 0.35 * rating_norm + 0.10 * fame_norm
    else:
        final = 0.65 * sim_norm + 0.20 * vote_priority + 0.15 * fame_norm

    # ── Exclude query movie itself ────────────────────────────────────────────
    if anchor_idx is not None:
        final[anchor_idx] = 0.0
        anchor_title_exact = str(df.loc[anchor_idx].get('title', '')).strip().lower()
        same_title_mask = (
            df['title'].fillna('').astype(str).str.strip().str.lower().values == anchor_title_exact
        )
        # Also remove duplicate rows of the exact same movie title.
        final[same_title_mask] = 0.0

    # ── Hard filters ─────────────────────────────────────────────────────────

    # Language filter
    lang_mask = np.ones(n, dtype=bool)
    if language_codes:
        lang_mask = df['language'].isin(language_codes).values
        final *= lang_mask

    # Decade filter
    dec_mask = _decade_mask(df, decade_filter)
    final   *= dec_mask

    # For pure genre-tag mode, enforce at least one selected genre overlap.
    if genre_only_mode and query_genres:
        tag_overlap = np.array([
            len(query_genres & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        final *= (tag_overlap >= 1)

    # Hard floor on vote_average: do not recommend movies rated below threshold.
    vote_avg_gate = (pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values >= min_vote_avg)
    final *= vote_avg_gate

    # Never recommend movies with zero votes.
    nonzero_vote_gate = (pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).values > 0)
    final *= nonzero_vote_gate

    # For movie-title anchored requests, keep only confident vote_count rows.
    vote_count_gate = np.ones(n, dtype=bool)
    if anchor_idx is not None:
        vote_count_gate = (
            (pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).values > 20) |
            (franchise_boost >= 0.75)
        )
        final *= vote_count_gate

    # Noisy title filter  
    clean_mask = np.array([_is_clean_title(t) for t in df['title']])
    final     *= clean_mask

    # ── Optional temporal window when anchor is available ────────────────────
    if anchor_idx is not None and q_year is not None:
        year_diff   = np.abs(df['release_year'].values - q_year)
        year_window = 15
        in_window   = (year_diff <= year_window)
        windowed    = final * in_window
        if int(windowed.astype(bool).sum()) >= top_n * 2:
            final = windowed

    # ── Genre hard filter when anchor genres are known ───────────────────────
    if anchor_idx is not None and q_genres_a:
        g_overlap_anchor = np.array([
            len(q_genres_a & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        genre_or_franchise_gate = (g_overlap_anchor > 2) | (franchise_boost >= 0.75)
        final *= genre_or_franchise_gate
        # Strong relevance nudge for same-franchise titles.
        final += 0.12 * franchise_boost

    # Keep existing broader genre alignment logic.
    if effective_genres and anchor_idx is not None:
        g_overlap = np.array([
            len(effective_genres & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        genre_ok = (g_overlap >= 1)
        genre_filtered = final * genre_ok
        if int(genre_filtered.astype(bool).sum()) >= top_n * 2:
            final = genre_filtered

    # ── Pick top_n candidates ─────────────────────────────────────────────────
    fetch_n  = top_n * 3 if diversify else top_n * 2
    top_idxs = np.argsort(final)[::-1][:fetch_n]
    top_idxs = top_idxs[final[top_idxs] > 0.0]

    # ── MMR diversification ───────────────────────────────────────────────────
    if diversify and _sbert_vecs is not None and len(top_idxs) > top_n:
        selected = _mmr(top_idxs, final, top_n)
    else:
        selected = top_idxs[:top_n]

    # ── Build result objects ──────────────────────────────────────────────────
    results = []
    for idx in selected:
        if idx >= len(df):
            continue
        row   = df.iloc[idx]
        score = float(final[idx])
        fm    = float(fame_norm[idx])

        cand_genres = set(row.get('genres', []) if isinstance(row.get('genres', []), list) else [])
        cand_keywords = set(row.get('keywords', []) if isinstance(row.get('keywords', []), list) else [])
        cand_cast = set((row.get('cast', []) or [])[:5]) if isinstance(row.get('cast', []), list) else set()
        shared_genres = sorted(list(effective_genres & cand_genres)) if effective_genres else []
        shared_keywords = sorted(list(q_keywords_a & cand_keywords)) if q_keywords_a else []
        shared_cast = sorted(list(q_cast_a & cand_cast)) if q_cast_a else []

        tfidf_val = float(s_tfidf[idx]) if idx < len(s_tfidf) else 0.0
        sbert_val = float(s_sbert[idx]) if idx < len(s_sbert) else 0.0
        cf_val = float(s_cf[idx]) if idx < len(s_cf) else 0.0
        genre_bonus_val = float(genre_bonus[idx]) if idx < len(genre_bonus) else 0.0
        total_sim_val = float(total_sim[idx]) if idx < len(total_sim) else 0.0
        sim_norm_val = float(sim_norm[idx]) if idx < len(sim_norm) else 0.0
        vote_priority_val = float(vote_priority[idx]) if idx < len(vote_priority) else 0.0
        rating_norm_val = float(rating_norm[idx]) if idx < len(rating_norm) else 0.0
        franchise_val = float(franchise_boost[idx]) if idx < len(franchise_boost) else 0.0

        if genre_only_mode:
            contrib = {
                'vote_priority(0.55)': 0.55 * vote_priority_val,
                'rating_norm(0.35)': 0.35 * rating_norm_val,
                'fame(0.10)': 0.10 * fm,
            }
        else:
            contrib = {
                'sim_norm(0.65)': 0.65 * sim_norm_val,
                'vote_priority(0.20)': 0.20 * vote_priority_val,
                'fame(0.15)': 0.15 * fm,
            }

        top_reasons = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]

        print(f"\n[DEBUG][HYBRID][IDX {idx}] {row.get('title', '')}")
        print(
            f"  final_score={score:.4f} | vote_avg={float(row.get('vote_average', 0.0)):.2f} | "
            f"vote_count={int(row.get('vote_count', 0))}"
        )
        print(
            f"  base signals: tfidf={tfidf_val:.4f}, sbert={sbert_val:.4f}, cf={cf_val:.4f}, "
            f"genre_bonus={genre_bonus_val:.4f}, total_sim={total_sim_val:.4f}"
        )
        print(
            f"  fused signals: sim_norm={sim_norm_val:.4f}, vote_priority={vote_priority_val:.4f}, "
            f"fame={fm:.4f}, rating_norm={rating_norm_val:.4f}, franchise_boost={franchise_val:.4f}"
        )
        print(
            "  weighted contribution: "
            + ", ".join(f"{k}={v:.4f}" for k, v in contrib.items())
        )
        print(
            "  why recommended: "
            + ", ".join(f"{k}={v:.4f}" for k, v in top_reasons)
        )
        print(
            "  gates: "
            f"lang={bool(lang_mask[idx])}, decade={bool(dec_mask[idx])}, "
            f"vote_avg={bool(vote_avg_gate[idx])}, nonzero_votes={bool(nonzero_vote_gate[idx])}, vote_count={bool(vote_count_gate[idx])}, "
            f"clean_title={bool(clean_mask[idx])}"
        )
        print(f"  genre         : {_fmt_list(row.get('genres', []), max_items=10)}")
        print(f"  keyword       : {_fmt_list(row.get('keywords', []), max_items=12)}")
        print(f"  cast          : {_fmt_list(row.get('cast', []), max_items=8)}")
        print(f"  shared_genres : {_fmt_list(shared_genres, max_items=10)}")
        print(f"  shared_keywords: {_fmt_list(shared_keywords, max_items=12)}")
        print(f"  shared_cast   : {_fmt_list(shared_cast, max_items=8)}")

        results.append(RecommendedMovie(
            movie_id       = str(row.get('id', idx)),
            title          = str(row.get('title', '')),
            original_title = str(row.get('original_title', '')),
            language       = str(row.get('language', '')),
            year           = int(row.get('release_year', 0)),
            runtime        = int(row.get('runtime', 0)),
            vote_average   = float(row.get('vote_average', 0)),
            vote_count     = int(row.get('vote_count', 0)),
            genres         = list(row.get('genres', []) or []),
            director       = str(row.get('director', '')),
            cast           = list(row.get('cast', []) or []),
            poster_path    = str(row.get('poster_path', '')),
            tagline        = str(row.get('tagline', '')),
            overview       = str(row.get('overview', '')),
            budget         = int(row.get('budget', 0)),
            revenue        = int(row.get('revenue', 0)),
            weighted_score = min(score, 1.0),
            fame_score     = fm,
        ))

    return results


def _mmr(candidates: np.ndarray, scores: np.ndarray, top_n: int,
          lambda_: float = 0.7) -> np.ndarray:
    """
    Maximal Marginal Relevance (MMR) diversification.
    lambda_ = relevance weight (higher → closer to pure ranking).
    """
    if _sbert_vecs is None:
        return candidates[:top_n]

    selected = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        if not selected:
            # Pick highest-scoring first
            best = max(remaining, key=lambda i: scores[i])
        else:
            sel_vecs = _sbert_vecs[selected]
            def mmr_score(i):
                rel  = scores[i]
                div  = float((_sbert_vecs[i] @ sel_vecs.T).max())
                return lambda_ * rel - (1 - lambda_) * div
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return np.array(selected)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_recommendations(
    collection,                        # ChromaDB collection (not used directly, kept for compat)
    movie_title:    Optional[str],
    free_text:      Optional[str],
    selected_chips: list,
    language_codes: list,
    top_n:          int  = 10,
    min_rating:     float = 5.0,       # kept for UI compat; treated as min vote_average
    decade_filter:  list = None,
    include_old_movies: bool = False,
    diversify:      bool = False,
    df:             Optional[pd.DataFrame] = None,  # passed from app if needed
) -> tuple[list[RecommendedMovie], str]:
    """
    Main entry point called by app.py.

    Returns (results, query_summary_string).
    """
    global _df

    # If engine not yet built but df is provided, build it now
    if not _engine_ready:
        if df is not None:
            build_engine(df)
        else:
            raise RuntimeError("Recommendation engine not yet built; call build_engine(df) first.")

    if decade_filter is None:
        decade_filter = ["2020s", "2010s", "2000s", "1990s", "Classic (<1990)"]

    # ── Resolve anchor ────────────────────────────────────────────────────────
    anchor_idx   = None
    anchor_label = None
    lang_hint    = language_codes[0] if len(language_codes) == 1 else None

    if movie_title:
        anchor_idx = _find_movie_idx(movie_title, language=lang_hint)
        if anchor_idx is not None:
            anchor_label = _df.loc[anchor_idx, 'title']

    # ── Build text query ──────────────────────────────────────────────────────
    query_text   = _build_query_text(free_text or '', selected_chips or [])

    # Query genre set from chips
    supported_genres = {
        "Action","Romance","Thriller","Drama","Comedy",
        "Horror","Family","Historical","Crime","Sci-Fi"
    }
    query_genres = {c for c in (selected_chips or []) if c in supported_genres}

    # Enforce minimum vote_average floor. If caller sends lower value, keep 5.0.
    min_vote_avg = max(5.0, float(min_rating))

    # If user selects only genre tags (no movie title, no free text), rank mostly
    # by vote_count and vote_average within matching genres.
    genre_only_mode = bool(
        anchor_idx is None and
        not (free_text or '').strip() and
        bool(selected_chips) and
        all(c in supported_genres for c in (selected_chips or []))
    )

    # ── Routing: title-only uses semantic recommender ────────────────────────
    title_only_mode = bool(anchor_idx is not None and not (free_text or '').strip() and not (selected_chips or []))

    if title_only_mode:
        results = _semantic_recommend_from_anchor(
            anchor_idx      = anchor_idx,
            language_codes  = language_codes,
            decade_filter   = decade_filter,
            top_n           = top_n,
            min_vote_avg    = min_vote_avg,
            year_window     = 12,
            min_genre_overlap = 2,
            min_vote_count  = 20,
        )

        # Fallback to hybrid if SBERT is unavailable or filters are too strict.
        if not results:
            results = _hybrid_recommend(
                anchor_idx     = anchor_idx,
                query_text     = query_text,
                query_genres   = query_genres,
                language_codes = language_codes,
                decade_filter  = decade_filter,
                top_n          = top_n,
                min_vote_avg   = min_vote_avg,
                genre_only_mode = False,
                diversify      = diversify,
            )
    else:
        results = _hybrid_recommend(
            anchor_idx     = anchor_idx,
            query_text     = query_text,
            query_genres   = query_genres,
            language_codes = language_codes,
            decade_filter  = decade_filter,
            top_n          = top_n,
            min_vote_avg   = min_vote_avg,
            genre_only_mode = genre_only_mode,
            diversify      = diversify,
        )

    # ── Build human-readable query summary ────────────────────────────────────
    parts = []
    if anchor_label:
        parts.append(f'Similar to \u201c{anchor_label}\u201d')
    if free_text:
        parts.append(free_text[:60] + ("…" if len(free_text) > 60 else ""))
    if selected_chips:
        parts.append(" · ".join(selected_chips))
    query_summary = "  |  ".join(parts) if parts else "Custom query"

    return results, query_summary


def sort_results(results: list[RecommendedMovie], sort_by: str) -> list[RecommendedMovie]:
    """Sort results by the chosen criterion."""
    if sort_by == "rating":
        return sorted(results, key=lambda m: m.vote_average, reverse=True)
    elif sort_by == "popularity":
        # Popularity sort now follows vote_count priority.
        return sorted(results, key=lambda m: m.vote_count, reverse=True)
    elif sort_by == "newest":
        return sorted(results, key=lambda m: m.year, reverse=True)
    else:  # best_match
        return sorted(results, key=lambda m: m.weighted_score, reverse=True)
