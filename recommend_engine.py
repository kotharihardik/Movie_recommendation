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
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _is_clean_title(x: str) -> bool:
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


# ── Hybrid recommender (core) ─────────────────────────────────────────────────

def _hybrid_recommend(
    anchor_idx:      Optional[int],
    query_text:      str,
    query_genres:    set,
    language_codes:  list,
    decade_filter:   list,
    top_n:           int,
    min_vote_avg:    float = 5.0,    # do not recommend movies rated below this
    diversify:       bool = False,
) -> list[RecommendedMovie]:
    """
    Core hybrid engine.

    Score = w1*tfidf + w2*sbert + w3*cf + fame_boost
    Weights depend on which signals are available.
    """
    df = _df
    n  = len(df)

    # ── Base similarity scores ────────────────────────────────────────────────
    if anchor_idx is not None:
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
        q_year     = int(q_row.get('release_year', 2000))

    else:
        # No anchor movie → pure text/chip query
        s_sbert     = _sbert_scores_from_text(query_text) if query_text else np.zeros(n)
        s_cf        = np.zeros(n)
        s_tfidf     = np.zeros(n)
        total_sim   = s_sbert
        q_genres_a  = set()
        q_year      = None

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
    final     = 0.65 * sim_norm + 0.20 * vote_priority + 0.15 * fame_norm

    # ── Exclude query movie itself ────────────────────────────────────────────
    if anchor_idx is not None:
        final[anchor_idx] = 0.0

    # ── Hard filters ─────────────────────────────────────────────────────────

    # Language filter
    if language_codes:
        lang_mask = df['language'].isin(language_codes).values
        final *= lang_mask

    # Decade filter
    dec_mask = _decade_mask(df, decade_filter)
    final   *= dec_mask

    # Hard floor on vote_average: do not recommend movies rated below threshold.
    vote_avg_gate = (pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values >= min_vote_avg)
    final *= vote_avg_gate

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

    # ── Genre hard filter when anchor and genres are known ───────────────────
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
    query_genres = {c for c in (selected_chips or []) if c in {
        "Action","Romance","Thriller","Drama","Comedy",
        "Horror","Family","Historical","Crime","Sci-Fi"
    }}

    # Enforce minimum vote_average floor. If caller sends lower value, keep 5.0.
    min_vote_avg = max(5.0, float(min_rating))

    # ── Run hybrid engine ─────────────────────────────────────────────────────
    results = _hybrid_recommend(
        anchor_idx     = anchor_idx,
        query_text     = query_text,
        query_genres   = query_genres,
        language_codes = language_codes,
        decade_filter  = decade_filter,
        top_n          = top_n,
        min_vote_avg   = min_vote_avg,
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
