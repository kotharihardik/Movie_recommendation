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

vote_count / vote_average are used only as light reliability priors
(vote_confidence + rating_norm); semantic relevance remains dominant.
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
_vote_confidence_scores: Optional[np.ndarray] = None   # aligned with _df index

SUPPORTED_GENRES = {
    "Action", "Romance", "Thriller", "Drama", "Comedy",
    "Horror", "Family", "Historical", "Crime", "Sci-Fi",
}

_GENERIC_THEME_KEYWORDS = {
    "love", "romance", "action", "drama", "comedy", "thriller",
    "family", "friendship", "fight", "hero", "villain", "movie",
}

SEMANTIC_SCORE_WEIGHTS = {
    "anchor_sim_rank": 0.40,
    "genre_jaccard": 0.14,
    "cast_jaccard": 0.22,
    "keyword_jaccard": 0.05,
    "temporal_soft": 0.05,
    "vote_confidence": 0.05,
    "rating_norm": 0.03,
    "fame_score": 0.04,
    "franchise_boost": 0.02,
}

ANCHOR_SIGNAL_WEIGHTS = {
    "tfidf": 0.35,
    "sbert": 0.60,
    "cf": 0.05,
}

ANCHOR_QUERY_TEXT_SBERT_BLEND = 0.20

SEMANTIC_SBERT_THEME_GATE = 0.80
SEMANTIC_CAST_IMMUNITY_JACCARD = 0.40


def _fmt_list(values, max_items: int = 8) -> str:
    """Format list-like values for concise terminal debug output."""
    if not isinstance(values, list) or not values:
        return "-"
    shown = values[:max_items]
    extra = "" if len(values) <= max_items else f" ... (+{len(values) - max_items} more)"
    return ", ".join(str(v) for v in shown) + extra


def _fmt_preview_text(text: str, max_chars: int = 220) -> str:
    """Single-line preview helper for potentially long text values."""
    t = re.sub(r'\s+', ' ', str(text or '')).strip()
    if not t:
        return "-"
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _debug_movie_meta(label: str, row: pd.Series, include_overview: bool = False) -> None:
    """Print essential movie metadata for debugging recommendation decisions."""
    print(f"\\n[DEBUG] {label}")
    print(f"  title       : {row.get('title', '')}")
    print(f"  year/lang   : {row.get('release_year', 0)} / {row.get('language', '')}")
    print(f"  vote        : avg={float(row.get('vote_average', 0.0)):.2f}, count={int(row.get('vote_count', 0))}")
    print(f"  genres      : {_fmt_list(row.get('genres', []), max_items=10)}")
    print(f"  keywords    : {_fmt_list(row.get('keywords', []), max_items=12)}")
    print(f"  cast        : {_fmt_list(row.get('cast', []), max_items=8)}")
    if include_overview:
        print(f"  overview    : {_fmt_preview_text(row.get('overview', ''), max_chars=220)}")


def _safe_float(v, default: float = 0.0) -> float:
    """Convert values to float while mapping NaN/inf/None to default."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _normalise_term_set(values) -> set:
    """Normalise token-like list values to lowercase string set."""
    if not isinstance(values, list):
        return set()
    out = set()
    for v in values:
        t = str(v).strip().lower()
        if t:
            out.add(t)
    return out


def _meaningful_keywords(values) -> set:
    """Keep only non-trivial keywords for theme-level matching."""
    terms = _normalise_term_set(values)
    return {
        t for t in terms
        if len(t.replace(' ', '')) >= 4 and t not in _GENERIC_THEME_KEYWORDS
    }


def _percentile_rank_scores(values: np.ndarray) -> np.ndarray:
    """Convert score array to [0,1] percentile ranks for better separation."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if float(arr.max() - arr.min()) < 1e-12:
        return np.zeros(arr.size, dtype=float)

    return (
        pd.Series(arr)
        .rank(method='average', pct=True)
        .to_numpy(dtype=float)
    )


def _debug_stage_header(scope: str, title: str) -> None:
    """Print a clear section header for recommendation debug logs."""
    print(f"\n[DEBUG][{scope}] {'=' * 62}")
    print(f"[DEBUG][{scope}] {title}")


def _debug_filter_step(scope: str, step: str, before: int, after: int, note: str = "") -> None:
    """Print count changes for each filtering stage."""
    removed = max(0, before - after)
    kept_pct = (100.0 * after / before) if before > 0 else 0.0
    suffix = f" | {note}" if note else ""
    print(
        f"[DEBUG][{scope}][FILTER] {step:<28} "
        f"{before:5d} -> {after:5d} (removed={removed:4d}, kept={kept_pct:5.1f}%)"
        f"{suffix}"
    )


def _debug_weight_breakdown(scope: str, components: list[tuple[str, float, float]]) -> None:
    """Print ordered score contribution rows: raw x weight => weighted."""
    print(f"[DEBUG][{scope}] weighted score breakdown")
    for i, (name, raw_value, weight) in enumerate(components, start=1):
        contribution = weight * raw_value
        print(
            f"  {i:>2}. {name:<22} "
            f"raw={raw_value:.4f}  x  w={weight:.2f}  =>  {contribution:.4f}"
        )


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
    """
    Build SBERT description with heavy keyword emphasis.

    Structure: overview + repeated keyword block so high-information themes
    get stronger representation in embeddings.
    """
    overview = re.sub(r'\s+', ' ', str(row.get('overview', '') or '')).strip()
    keywords = row.get('keywords', []) or []
    kw_terms = [
        str(k).strip()
        for k in (keywords if isinstance(keywords, list) else [])
        if str(k).strip()
    ]
    kw_str = ' '.join(kw_terms)

    if overview and kw_str:
        return f"{overview}. Themes: {kw_str} {kw_str}".strip()
    if overview:
        return overview
    if kw_str:
        return f"Themes: {kw_str} {kw_str}".strip()
    return str(row.get('title', '') or '').strip()


# ── Popularity/reliability priors ────────────────────────────────────────────

def _compute_fame_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Star-power fame heuristic (does NOT use popularity column).

    Improvements:
      • Count actor/director appearances only from movies with vote_count > 50
      • Weight cast by billing position (top-billed contributes more)
      • Keep director as a secondary signal
    """
    vc = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).clip(lower=0)
    qualified_mask = vc > 50

    from collections import Counter
    actor_counts: Counter = Counter()
    dir_counts: Counter = Counter()

    for (_, row), is_qualified in zip(df.iterrows(), qualified_mask):
        if not bool(is_qualified):
            continue
        director = str(row.get('director', '') or '')
        if director:
            dir_counts[director] += 1
        cast_list = [a for a in (row.get('cast', []) or [])][:5]
        for actor in cast_list:
            actor_counts[actor] += 1

    raw_fame = np.zeros(len(df), dtype=float)
    pos_weights = [1.0, 0.7, 0.5]  # top-billed cast gets highest contribution

    for i, (_, row) in enumerate(df.iterrows()):
        cast_list = [a for a in (row.get('cast', []) or [])][:3]
        cast_term = 0.0
        for pos, actor in enumerate(cast_list):
            w = pos_weights[pos] if pos < len(pos_weights) else 0.2
            cast_term += w * math.log1p(actor_counts.get(actor, 0))

        director = str(row.get('director', '') or '')
        director_term = 0.45 * math.log1p(dir_counts.get(director, 0))
        raw_fame[i] = cast_term + director_term

    scaler = MinMaxScaler()
    return scaler.fit_transform(raw_fame.reshape(-1, 1)).flatten()


def _compute_vote_confidence_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Build a vote-count confidence prior in [0, 1].

    This is a reliability signal (sample-size confidence), not a popularity
    rank: log scaling keeps huge vote_count outliers from dominating.
    """
    vc = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).clip(lower=0)
    vc_max = float(vc.max()) if len(vc) else 0.0
    if vc_max <= 0:
        return np.zeros(len(df), dtype=float)
    return (np.log1p(vc) / np.log1p(vc_max + 1e-9)).to_numpy(dtype=float)


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
    global _vote_confidence_scores

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

    # ── Fame + vote confidence scores ────────────────────────────────────────
    print("  [4/4] Computing fame and vote-confidence scores…")
    _fame_scores = _compute_fame_scores(_df)
    _vote_confidence_scores = _compute_vote_confidence_scores(_df)
    print(
        f"     Fame range: [{_fame_scores.min():.3f}, {_fame_scores.max():.3f}] | "
        f"Vote-confidence range: "
        f"[{_vote_confidence_scores.min():.3f}, {_vote_confidence_scores.max():.3f}]"
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


def _compute_anchor_similarity_bundle(anchor_idx: int, query_text: str = "") -> dict[str, np.ndarray]:
    """Reusable anchor similarity block shared by hybrid and title-only recommenders."""
    n = len(_df) if _df is not None else 0
    zeros = np.zeros(n, dtype=float)

    if _df is None or anchor_idx < 0 or anchor_idx >= n:
        return {
            "tfidf": zeros,
            "sbert": zeros,
            "cf": zeros,
            "total": zeros,
        }

    s_tfidf = _tfidf_scores(anchor_idx)
    s_sbert = _sbert_scores_from_idx(anchor_idx) if _sbert_vecs is not None else np.zeros(n, dtype=float)
    if query_text:
        s_text_extra = _sbert_scores_from_text(query_text)
        s_sbert = (
            (1.0 - ANCHOR_QUERY_TEXT_SBERT_BLEND) * s_sbert +
            ANCHOR_QUERY_TEXT_SBERT_BLEND * s_text_extra
        )
    s_cf = _knn_scores(anchor_idx)

    total_sim = (
        ANCHOR_SIGNAL_WEIGHTS['tfidf'] * s_tfidf +
        ANCHOR_SIGNAL_WEIGHTS['sbert'] * s_sbert +
        ANCHOR_SIGNAL_WEIGHTS['cf'] * s_cf
    )
    if anchor_idx < len(total_sim):
        total_sim[anchor_idx] = 0.0

    return {
        "tfidf": s_tfidf,
        "sbert": s_sbert,
        "cf": s_cf,
        "total": total_sim,
    }


def _attach_semantic_priors(frame: pd.DataFrame, base_df: pd.DataFrame) -> None:
    """Attach vote/fame/rating priors to a candidate frame aligned by index."""
    vote_conf_all = _vote_confidence_scores if _vote_confidence_scores is not None else np.zeros(len(base_df))
    fame_all = _fame_scores if _fame_scores is not None else np.zeros(len(base_df))

    frame['vote_confidence'] = pd.Series(vote_conf_all, index=base_df.index).reindex(frame.index).fillna(0.0)
    frame['fame_score'] = pd.Series(fame_all, index=base_df.index).reindex(frame.index).fillna(0.0)
    frame['rating_norm'] = (
        pd.to_numeric(frame['weighted_rating'], errors='coerce')
        .fillna(0)
        .clip(lower=0, upper=10) / 10.0
    )


def _compute_semantic_score(frame: pd.DataFrame) -> pd.Series:
    """Compute title-only semantic score using anchor similarity + structural priors."""
    return (
        SEMANTIC_SCORE_WEIGHTS['anchor_sim_rank'] * frame['anchor_sim_rank'] +
        SEMANTIC_SCORE_WEIGHTS['genre_jaccard'] * frame['genre_jaccard'] +
        SEMANTIC_SCORE_WEIGHTS['cast_jaccard'] * frame['cast_jaccard'] +
        SEMANTIC_SCORE_WEIGHTS['keyword_jaccard'] * frame['keyword_jaccard'] +
        SEMANTIC_SCORE_WEIGHTS['temporal_soft'] * frame['temporal_soft'] +
        SEMANTIC_SCORE_WEIGHTS['vote_confidence'] * frame['vote_confidence'] +
        SEMANTIC_SCORE_WEIGHTS['rating_norm'] * frame['rating_norm'] +
        SEMANTIC_SCORE_WEIGHTS['fame_score'] * frame['fame_score'] +
        SEMANTIC_SCORE_WEIGHTS['franchise_boost'] * frame['franchise_boost']
    )


# ── Chip / free-text query builder ───────────────────────────────────────────

def _build_query_text(free_text: str, selected_chips: list) -> str:
    """Merge free-text + non-genre chips into one SBERT query string."""
    parts = []
    if selected_chips:
        # Important: do not inject genre chips into SBERT text query.
        # Genres are used as structural filters elsewhere.
        moods = [c for c in selected_chips if c not in SUPPORTED_GENRES]
        if moods:
            parts.append(f"The mood is {', '.join(moods).lower()}.")
    if free_text:
        parts.append(free_text.strip())
    return ' '.join(parts).strip()


def _decade_label_from_year(year: int) -> str:
    """Map release year to UI decade label."""
    if year >= 2020:
        return "2020s"
    if year >= 2010:
        return "2010s"
    if year >= 2000:
        return "2000s"
    if year >= 1990:
        return "1990s"
    return "Classic (<1990)"


def _ensure_anchor_decade(decade_filter: list, anchor_year: int) -> list:
    """Ensure the anchor movie's decade is included in decade filters."""
    if not decade_filter:
        return decade_filter

    updated = list(decade_filter)
    anchor_decade = _decade_label_from_year(anchor_year)

    if anchor_decade == "Classic (<1990)":
        has_classic = any("Classic" in str(d) for d in updated)
        if not has_classic:
            updated.append(anchor_decade)
        return updated

    if anchor_decade not in updated:
        updated.append(anchor_decade)
    return updated


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
    anchor_base = _normalise_title_for_franchise(anchor_title)
    if not anchor_base:
        return np.zeros(len(df), dtype=float)

    anchor_tokens = set(anchor_base.split())
    scores = np.zeros(len(df), dtype=float)

    for i, t in enumerate(df['title'].fillna('').astype(str).tolist()):
        cand_base = _normalise_title_for_franchise(t)
        if not cand_base:
            continue
        cand_tokens = set(cand_base.split())
        # Subset containment (anchor ⊆ candidate) gives 1.0
        if anchor_tokens.issubset(cand_tokens):
            scores[i] = 1.0
        elif cand_tokens.issubset(anchor_tokens):
            scores[i] = 0.95   # e.g., "Phir Hera Pheri" -> "Hera Pheri"
        else:
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
    Uses blended anchor similarity (TF-IDF + SBERT + CF) with overlap/time/quality guardrails.
    """
    if _df is None:
        return []

    df = _df
    q_row = df.loc[anchor_idx]
    anchor_title_exact = str(q_row.get('title', '')).strip().lower()

    anchor_sims = _compute_anchor_similarity_bundle(anchor_idx=anchor_idx, query_text="")

    q_genres = set(q_row['genres']) if isinstance(q_row.get('genres'), list) else set()
    q_keywords = _meaningful_keywords(q_row.get('keywords', []))
    q_cast = set((q_row.get('cast') or [])[:5]) if isinstance(q_row.get('cast'), list) else set()
    q_year = int(q_row.get('release_year', 2000))

    _debug_stage_header("SEMANTIC", "Title-only semantic recommendation")
    _debug_movie_meta("Input movie", q_row, include_overview=True)

    candidate_df = df.copy()
    _debug_filter_step("SEMANTIC", "start candidates", len(df), len(candidate_df))
    candidate_df['tfidf_raw'] = pd.Series(anchor_sims['tfidf'], index=df.index)
    candidate_df['sbert_raw'] = pd.Series(anchor_sims['sbert'], index=df.index)
    candidate_df['cf_raw'] = pd.Series(anchor_sims['cf'], index=df.index)
    candidate_df['anchor_sim_raw'] = pd.Series(anchor_sims['total'], index=df.index)
    franchise_scores = _franchise_boost_scores(df, str(q_row.get('title', '')))
    candidate_df['franchise_boost'] = pd.Series(franchise_scores, index=df.index)

    before = len(candidate_df)
    candidate_df = candidate_df[candidate_df['title'].apply(_is_clean_title)]
    _debug_filter_step("SEMANTIC", "clean title", before, len(candidate_df))

    # Never recommend the exact same movie title in title-search mode.
    before = len(candidate_df)
    candidate_df = candidate_df[
        candidate_df['title'].fillna('').astype(str).str.strip().str.lower() != anchor_title_exact
    ]
    _debug_filter_step("SEMANTIC", "exclude same title", before, len(candidate_df))

    candidate_df['genre_overlap'] = candidate_df['genres'].apply(
        lambda g: len(q_genres & set(g)) if isinstance(g, list) else 0
    )
    candidate_df['genre_jaccard'] = candidate_df['genres'].apply(
        lambda g: _jaccard(q_genres, set(g)) if isinstance(g, list) else 0.0
    )
    candidate_df['cast_jaccard'] = candidate_df['cast'].apply(
        lambda c: _jaccard(q_cast, set(c[:5])) if isinstance(c, list) else 0.0
    )

    # Title-mode structural filter: require minimum genre overlap, but keep
    # high cast-overlap candidates (sequel/cast continuity immunity).
    required_genre_overlap = min(min_genre_overlap, len(q_genres)) if q_genres else 0
    if q_genres:
        before = len(candidate_df)
        candidate_df = candidate_df[
            (candidate_df['genre_overlap'] >= required_genre_overlap) |
            (candidate_df['cast_jaccard'] >= SEMANTIC_CAST_IMMUNITY_JACCARD)
        ]
        _debug_filter_step(
            "SEMANTIC",
            "genre overlap gate",
            before,
            len(candidate_df),
            note=(
                f"required_overlap>={required_genre_overlap} "
                f"or cast_jaccard>={SEMANTIC_CAST_IMMUNITY_JACCARD:.2f}"
            ),
        )

    if 'keywords' in candidate_df.columns:
        candidate_df['keyword_overlap'] = candidate_df['keywords'].apply(
            lambda k: len(q_keywords & _meaningful_keywords(k)) if isinstance(k, list) else 0
        )
        candidate_df['keyword_jaccard'] = candidate_df['keywords'].apply(
            lambda k: _jaccard(q_keywords, _meaningful_keywords(k)) if isinstance(k, list) else 0.0
        )
    else:
        candidate_df['keyword_overlap'] = 0
        candidate_df['keyword_jaccard'] = 0.0

    if q_keywords:
        before = len(candidate_df)
        candidate_df = candidate_df[
            (candidate_df['keyword_overlap'] >= 1) |
            (candidate_df['sbert_raw'] >= SEMANTIC_SBERT_THEME_GATE) |
            (candidate_df['franchise_boost'] >= 0.75)
        ]
        _debug_filter_step(
            "SEMANTIC",
            "theme keyword/SBERT gate",
            before,
            len(candidate_df),
            note=f"keyword_overlap>=1 or sbert>={SEMANTIC_SBERT_THEME_GATE:.2f}",
        )

    year_diff = np.abs(candidate_df['release_year'].astype(float) - float(q_year))
    sigma = float(max(year_window, 1))
    candidate_df['temporal_soft'] = np.exp(-(year_diff ** 2) / (2.0 * (sigma ** 2)))

    candidate_df['anchor_sim_rank'] = _percentile_rank_scores(
        pd.to_numeric(candidate_df['anchor_sim_raw'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    )
    candidate_df['sbert_rank'] = _percentile_rank_scores(
        pd.to_numeric(candidate_df['sbert_raw'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    )

    _attach_semantic_priors(candidate_df, df)
    candidate_df['semantic_score'] = _compute_semantic_score(candidate_df)

    if anchor_idx in candidate_df.index:
        candidate_df.loc[anchor_idx, 'semantic_score'] = 0.0

    if language_codes:
        before = len(candidate_df)
        candidate_df = candidate_df[candidate_df['language'].isin(language_codes)]
        _debug_filter_step(
            "SEMANTIC",
            "language filter",
            before,
            len(candidate_df),
            note=f"languages={language_codes}",
        )

    dec_mask = _decade_mask(candidate_df, decade_filter)
    before = len(candidate_df)
    candidate_df = candidate_df[dec_mask]
    _debug_filter_step(
        "SEMANTIC",
        "decade filter",
        before,
        len(candidate_df),
        note=f"decades={decade_filter}",
    )

    before = len(candidate_df)
    candidate_df = candidate_df[
        (pd.to_numeric(candidate_df['vote_average'], errors='coerce').fillna(0) >= min_vote_avg) |
        (candidate_df['franchise_boost'] >= 0.75)
    ]
    _debug_filter_step(
        "SEMANTIC",
        "vote_average gate",
        before,
        len(candidate_df),
        note=f"vote_avg>={min_vote_avg:.1f} or franchise>=0.75",
    )

    # Never recommend movies with zero votes.
    before = len(candidate_df)
    candidate_df = candidate_df[
        pd.to_numeric(candidate_df['vote_count'], errors='coerce').fillna(0) > 0
    ]
    _debug_filter_step("SEMANTIC", "non-zero votes", before, len(candidate_df))

    # Hard vote-count floor for movie-title mode.
    before = len(candidate_df)
    candidate_df = candidate_df[
        (pd.to_numeric(candidate_df['vote_count'], errors='coerce').fillna(0) > min_vote_count) |
        (candidate_df['franchise_boost'] >= 0.75)
    ]
    _debug_filter_step(
        "SEMANTIC",
        "vote_count floor",
        before,
        len(candidate_df),
        note=f"vote_count>{min_vote_count} or franchise>=0.75",
    )

    in_window = candidate_df[
        np.abs(pd.to_numeric(candidate_df['release_year'], errors='coerce').fillna(0) - q_year) <= year_window
    ]
    if len(in_window) >= top_n:
        _debug_filter_step(
            "SEMANTIC",
            "year window",
            len(candidate_df),
            len(in_window),
            note=f"|year-query_year|<={year_window}",
        )
        candidate_df = in_window

    if len(candidate_df) < top_n:
        print(
            f"[DEBUG][SEMANTIC] candidate count {len(candidate_df)} < top_n {top_n}; "
            "running relaxed fill stage"
        )
        relaxed_df = df.copy()
        relaxed_df = relaxed_df[relaxed_df['title'].apply(_is_clean_title)]
        relaxed_df = relaxed_df[
            relaxed_df['title'].fillna('').astype(str).str.strip().str.lower() != anchor_title_exact
        ]
        relaxed_df['tfidf_raw'] = pd.Series(anchor_sims['tfidf'], index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['sbert_raw'] = pd.Series(anchor_sims['sbert'], index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['cf_raw'] = pd.Series(anchor_sims['cf'], index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['anchor_sim_raw'] = pd.Series(anchor_sims['total'], index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['franchise_boost'] = pd.Series(franchise_scores, index=df.index).reindex(relaxed_df.index).fillna(0.0)
        relaxed_df['genre_jaccard'] = relaxed_df['genres'].apply(
            lambda g: _jaccard(q_genres, set(g)) if isinstance(g, list) else 0.0
        )
        relaxed_df['cast_jaccard'] = relaxed_df['cast'].apply(
            lambda c: _jaccard(q_cast, set(c[:5])) if isinstance(c, list) else 0.0
        )
        if 'keywords' in relaxed_df.columns:
            relaxed_df['keyword_overlap'] = relaxed_df['keywords'].apply(
                lambda k: len(q_keywords & _meaningful_keywords(k)) if isinstance(k, list) else 0
            )
            relaxed_df['keyword_jaccard'] = relaxed_df['keywords'].apply(
                lambda k: _jaccard(q_keywords, _meaningful_keywords(k)) if isinstance(k, list) else 0.0
            )
        else:
            relaxed_df['keyword_overlap'] = 0
            relaxed_df['keyword_jaccard'] = 0.0

        if q_keywords:
            relaxed_df = relaxed_df[
                (relaxed_df['keyword_overlap'] >= 1) |
                (relaxed_df['sbert_raw'] >= SEMANTIC_SBERT_THEME_GATE) |
                (relaxed_df['franchise_boost'] >= 0.75)
            ]

        relaxed_year_diff = np.abs(
            pd.to_numeric(relaxed_df['release_year'], errors='coerce').fillna(0) - q_year
        )
        relaxed_df['temporal_soft'] = np.exp(-(relaxed_year_diff ** 2) / (2.0 * (float(max(year_window, 1)) ** 2)))
        relaxed_df['anchor_sim_rank'] = _percentile_rank_scores(
            pd.to_numeric(relaxed_df['anchor_sim_raw'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        )
        relaxed_df['sbert_rank'] = _percentile_rank_scores(
            pd.to_numeric(relaxed_df['sbert_raw'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        )
        _attach_semantic_priors(relaxed_df, df)
        relaxed_df['semantic_score'] = _compute_semantic_score(relaxed_df)
        if anchor_idx in relaxed_df.index:
            relaxed_df.loc[anchor_idx, 'semantic_score'] = 0.0
        if language_codes:
            relaxed_df = relaxed_df[relaxed_df['language'].isin(language_codes)]
        relaxed_dec_mask = _decade_mask(relaxed_df, decade_filter)
        relaxed_df = relaxed_df[relaxed_dec_mask]
        if q_genres:
            # Fill-stage genre filter with cast immunity to avoid dropping sequels
            # due to inconsistent genre tagging.
            relaxed_df = relaxed_df[
                (
                    relaxed_df['genres'].apply(
                        lambda g: len(q_genres & set(g)) if isinstance(g, list) else 0
                    ) >= required_genre_overlap
                ) |
                (relaxed_df['cast_jaccard'] >= SEMANTIC_CAST_IMMUNITY_JACCARD)
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
        before = len(candidate_df)
        candidate_df = pd.concat([candidate_df, relaxed_df], axis=0)
        candidate_df = candidate_df[~candidate_df.index.duplicated(keep='first')]
        _debug_filter_step("SEMANTIC", "after relaxed merge", before, len(candidate_df))

    # Ensure numeric dtypes for robust ranking with nlargest.
    candidate_df['semantic_score'] = pd.to_numeric(
        candidate_df.get('semantic_score', 0.0), errors='coerce'
    ).fillna(0.0)
    candidate_df['franchise_boost'] = pd.to_numeric(
        candidate_df.get('franchise_boost', 0.0), errors='coerce'
    ).fillna(0.0)
    for col in [
        'tfidf_raw',
        'sbert_raw',
        'cf_raw',
        'anchor_sim_raw',
        'anchor_sim_rank',
        'sbert_rank',
        'genre_jaccard',
        'cast_jaccard',
        'keyword_overlap',
        'keyword_jaccard',
        'temporal_soft',
        'vote_confidence',
        'rating_norm',
        'fame_score',
    ]:
        candidate_df[col] = pd.to_numeric(candidate_df.get(col, 0.0), errors='coerce').fillna(0.0)

    # Final ranking: reserve slots for strong franchise matches first, then fill.
    # This keeps sequel/series continuity without hardcoding any specific movie.
    franchise_quota = min(3, top_n)
    franchise_df = candidate_df[candidate_df['franchise_boost'] >= 0.90].sort_values(
        by=['semantic_score', 'vote_confidence', 'fame_score', 'vote_count'],
        ascending=[False, False, False, False],
    ).head(franchise_quota)
    remaining_df = candidate_df.drop(index=franchise_df.index, errors='ignore').sort_values(
        by=['semantic_score', 'vote_confidence', 'fame_score', 'vote_count'],
        ascending=[False, False, False, False],
    )
    top = pd.concat([franchise_df, remaining_df], axis=0).head(top_n)
    print(
        f"[DEBUG][SEMANTIC] final shortlist: {len(top)} "
        f"(franchise_priority={len(franchise_df)}, remaining={len(top) - len(franchise_df)})"
    )
    fame_norm = _fame_scores if _fame_scores is not None else np.zeros(len(df))

    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        cand_genres = set(row.get('genres', []) if isinstance(row.get('genres', []), list) else [])
        cand_keywords = _meaningful_keywords(row.get('keywords', []))
        cand_cast = set((row.get('cast', []) or [])[:5]) if isinstance(row.get('cast', []), list) else set()

        shared_genres = sorted(list(q_genres & cand_genres))
        shared_keywords = sorted(list(q_keywords & cand_keywords))
        shared_cast = sorted(list(q_cast & cand_cast))

        tfidf_raw = _safe_float(row.get('tfidf_raw', 0.0))
        sbert_raw = _safe_float(row.get('sbert_raw', 0.0))
        cf_raw = _safe_float(row.get('cf_raw', 0.0))
        anchor_sim_raw = _safe_float(row.get('anchor_sim_raw', 0.0))
        anchor_sim_rank = _safe_float(row.get('anchor_sim_rank', 0.0))
        sbert_rank = _safe_float(row.get('sbert_rank', 0.0))
        genre_j = _safe_float(row.get('genre_jaccard', 0.0))
        cast_j = _safe_float(row.get('cast_jaccard', 0.0))
        keyword_overlap = int(_safe_float(row.get('keyword_overlap', 0.0)))
        keyword_j = _safe_float(row.get('keyword_jaccard', 0.0))
        temporal = _safe_float(row.get('temporal_soft', 0.0))
        vote_confidence = _safe_float(row.get('vote_confidence', 0.0))
        rating_norm = _safe_float(row.get('rating_norm', 0.0))
        fame_signal = _safe_float(row.get('fame_score', 0.0))
        franchise = _safe_float(row.get('franchise_boost', 0.0))
        semantic = _safe_float(row.get('semantic_score', 0.0))

        components = [
            ("anchor_sim_rank", anchor_sim_rank, SEMANTIC_SCORE_WEIGHTS['anchor_sim_rank']),
            ("genre_jaccard", genre_j, SEMANTIC_SCORE_WEIGHTS['genre_jaccard']),
            ("cast_jaccard", cast_j, SEMANTIC_SCORE_WEIGHTS['cast_jaccard']),
            ("keyword_jaccard", keyword_j, SEMANTIC_SCORE_WEIGHTS['keyword_jaccard']),
            ("temporal_soft", temporal, SEMANTIC_SCORE_WEIGHTS['temporal_soft']),
            ("vote_confidence", vote_confidence, SEMANTIC_SCORE_WEIGHTS['vote_confidence']),
            ("rating_norm", rating_norm, SEMANTIC_SCORE_WEIGHTS['rating_norm']),
            ("fame_score", fame_signal, SEMANTIC_SCORE_WEIGHTS['fame_score']),
            ("franchise_boost", franchise, SEMANTIC_SCORE_WEIGHTS['franchise_boost']),
        ]
        top_reasons = sorted(components, key=lambda x: x[1] * x[2], reverse=True)[:3]

        print(f"\n[DEBUG][SEMANTIC][RANK {rank:02d}] {row.get('title', '')} (idx={idx})")
        print(
            f"  final_semantic_score={semantic:.4f} | "
            f"vote_count={int(row.get('vote_count', 0))} | "
            f"vote_avg={float(row.get('vote_average', 0.0)):.2f}"
        )
        print(
            f"  semantic primitives: tfidf_raw={tfidf_raw:.4f}, sbert_raw={sbert_raw:.4f}, cf_raw={cf_raw:.4f}, "
            f"anchor_sim_raw={anchor_sim_raw:.4f}, anchor_sim_rank={anchor_sim_rank:.4f}, "
            f"keyword_overlap={keyword_overlap}"
        )
        _debug_weight_breakdown("SEMANTIC", components)
        print(
            "  why recommended: "
            + ", ".join(f"{name}={raw * wt:.4f}" for name, raw, wt in top_reasons)
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

    _debug_stage_header("HYBRID", "Hybrid recommendation pipeline")

    # ── Base similarity scores ────────────────────────────────────────────────
    if anchor_idx is not None:
        q_title = str(df.loc[anchor_idx].get('title', ''))
        franchise_boost = _franchise_boost_scores(df, q_title)

        anchor_sims = _compute_anchor_similarity_bundle(anchor_idx=anchor_idx, query_text=query_text)
        s_tfidf = anchor_sims['tfidf']
        s_sbert = anchor_sims['sbert']
        s_cf = anchor_sims['cf']
        total_sim = anchor_sims['total']

        # Anchor movie's genre set for structural overlap guardrail
        q_row      = df.loc[anchor_idx]
        q_genres_a = set(q_row['genres']) if isinstance(q_row.get('genres'), list) else set()
        q_keywords_a = _meaningful_keywords(q_row.get('keywords', []))
        q_cast_a = set((q_row.get('cast') or [])[:5]) if isinstance(q_row.get('cast'), list) else set()
        q_year     = int(q_row.get('release_year', 2000))

        _debug_movie_meta("Input movie", q_row, include_overview=True)
        print(
            f"[DEBUG][HYBRID] base mix: "
            f"tfidf*{ANCHOR_SIGNAL_WEIGHTS['tfidf']:.2f} + "
            f"sbert*{ANCHOR_SIGNAL_WEIGHTS['sbert']:.2f} + "
            f"cf*{ANCHOR_SIGNAL_WEIGHTS['cf']:.2f} | "
            f"text_boost={'on' if bool(query_text) else 'off'}"
        )

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

        print("[DEBUG] Input query (no anchor movie)")
        print(f"  query_text   : {query_text if query_text else '-'}")
        print(f"  query_genres : {_fmt_list(sorted(list(query_genres)), max_items=10)}")

    text_semantic_gate = np.ones(n, dtype=bool)
    if anchor_idx is None and query_text.strip():
        sbert_cutoff = float(np.quantile(s_sbert, 0.80)) if len(s_sbert) else 0.0
        text_semantic_gate = (s_sbert >= sbert_cutoff)
        if int(text_semantic_gate.sum()) < top_n:
            top_semantic_n = min(max(top_n, 1), n)
            forced = np.argsort(s_sbert)[::-1][:top_semantic_n]
            text_semantic_gate = np.zeros(n, dtype=bool)
            text_semantic_gate[forced] = True
        print(
            f"[DEBUG][HYBRID] text semantic hard-gate: raw_sbert >= {sbert_cutoff:.4f} "
            f"(kept={int(text_semantic_gate.sum())}/{n})"
        )

    # ── Genre overlap bonus (structural guardrail, from notebook) ─────────────
    effective_genres = query_genres | q_genres_a
    if effective_genres:
        genre_bonus = _genre_overlap_scores(df, effective_genres)
        total_sim  = 0.82 * total_sim + 0.18 * genre_bonus
        print(
            f"[DEBUG][HYBRID] applied genre overlap blend: 0.82*base + 0.18*genre_bonus "
            f"with effective_genres={sorted(list(effective_genres))}"
        )

    # ── Final ranking integration ─────────────────────────────────────────────
    # Semantic relevance first, then reliability/popularity priors.
    sim_norm = _percentile_rank_scores(total_sim)
    fame_norm = _fame_scores if _fame_scores is not None else np.zeros(n)
    vote_confidence = (
        _vote_confidence_scores
        if _vote_confidence_scores is not None else np.zeros(n)
    )

    rating_source = pd.to_numeric(df.get('weighted_rating', df['vote_average']), errors='coerce').fillna(0)
    rating_norm = (
        rating_source.clip(lower=0, upper=10).values / 10.0
    )

    if genre_only_mode:
        # For pure genre-tag queries, emphasize reliable/popular known movies.
        final = 0.55 * vote_confidence + 0.30 * fame_norm + 0.15 * rating_norm
        print("[DEBUG][HYBRID] final mix (genre_only_mode): vote_confidence*0.55 + fame*0.30 + rating_norm*0.15")
    else:
        final = 0.70 * sim_norm + 0.17 * vote_confidence + 0.10 * fame_norm + 0.03 * rating_norm
        print("[DEBUG][HYBRID] final mix: sim_norm*0.70 + vote_confidence*0.17 + fame*0.10 + rating_norm*0.03")

    def _count_positive(arr: np.ndarray) -> int:
        return int(np.count_nonzero(arr > 0.0))

    if anchor_idx is None and query_text.strip():
        before = _count_positive(final)
        final *= text_semantic_gate
        _debug_filter_step(
            "HYBRID",
            "text semantic hard-gate",
            before,
            _count_positive(final),
            note="kept top 20% raw SBERT cosine",
        )

    print(f"[DEBUG][HYBRID][FILTER] start positive candidates: {_count_positive(final)}")

    # ── Exclude query movie itself ────────────────────────────────────────────
    if anchor_idx is not None:
        final[anchor_idx] = 0.0
        anchor_title_exact = str(df.loc[anchor_idx].get('title', '')).strip().lower()
        same_title_mask = (
            df['title'].fillna('').astype(str).str.strip().str.lower().values == anchor_title_exact
        )
        # Also remove duplicate rows of the exact same movie title.
        final[same_title_mask] = 0.0
        print(f"[DEBUG][HYBRID][FILTER] after removing anchor/same-title rows: {_count_positive(final)}")

    # ── Hard filters ─────────────────────────────────────────────────────────

    # Language filter
    lang_mask = np.ones(n, dtype=bool)
    if language_codes:
        before = _count_positive(final)
        lang_mask = df['language'].isin(language_codes).values
        final *= lang_mask
        _debug_filter_step(
            "HYBRID",
            "language filter",
            before,
            _count_positive(final),
            note=f"languages={language_codes}",
        )

    # Decade filter
    dec_mask = _decade_mask(df, decade_filter)
    before = _count_positive(final)
    final   *= dec_mask
    _debug_filter_step(
        "HYBRID",
        "decade filter",
        before,
        _count_positive(final),
        note=f"decades={decade_filter}",
    )

    # For pure genre-tag mode, enforce at least one selected genre overlap.
    if genre_only_mode and query_genres:
        tag_overlap = np.array([
            len(query_genres & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        before = _count_positive(final)
        final *= (tag_overlap >= 1)
        _debug_filter_step(
            "HYBRID",
            "genre tag overlap",
            before,
            _count_positive(final),
            note="requires >=1 selected genre",
        )

    # Hard floor on vote_average: do not recommend movies rated below threshold.
    vote_avg_gate = (pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values >= min_vote_avg)
    before = _count_positive(final)
    final *= vote_avg_gate
    _debug_filter_step(
        "HYBRID",
        "vote_average gate",
        before,
        _count_positive(final),
        note=f"vote_avg>={min_vote_avg:.1f}",
    )

    # Never recommend movies with zero votes.
    nonzero_vote_gate = (pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).values > 0)
    before = _count_positive(final)
    final *= nonzero_vote_gate
    _debug_filter_step("HYBRID", "non-zero votes", before, _count_positive(final))

    # For movie-title anchored requests, keep only confident vote_count rows.
    vote_count_gate = np.ones(n, dtype=bool)
    if anchor_idx is not None:
        vote_count_gate = (
            (pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).values > 20) |
            (franchise_boost >= 0.75)
        )
        before = _count_positive(final)
        final *= vote_count_gate
        _debug_filter_step(
            "HYBRID",
            "vote_count floor",
            before,
            _count_positive(final),
            note="vote_count>20 or franchise>=0.75",
        )

    # Noisy title filter  
    clean_mask = np.array([_is_clean_title(t) for t in df['title']])
    before = _count_positive(final)
    final     *= clean_mask
    _debug_filter_step("HYBRID", "clean title", before, _count_positive(final))

    # ── Optional temporal window when anchor is available ────────────────────
    if anchor_idx is not None and q_year is not None:
        year_diff   = np.abs(df['release_year'].values - q_year)
        year_window = 15
        in_window   = (year_diff <= year_window)
        windowed    = final * in_window
        if int(windowed.astype(bool).sum()) >= top_n * 2:
            _debug_filter_step(
                "HYBRID",
                "year window",
                _count_positive(final),
                int(windowed.astype(bool).sum()),
                note=f"|year-query_year|<={year_window}",
            )
            final = windowed

    # ── Genre hard filter when anchor genres are known ───────────────────────
    if anchor_idx is not None and q_genres_a:
        g_overlap_anchor = np.array([
            len(q_genres_a & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        genre_or_franchise_gate = (g_overlap_anchor > 2) | (franchise_boost >= 0.75)
        before = _count_positive(final)
        final *= genre_or_franchise_gate
        # Strong relevance nudge for same-franchise titles.
        final += 0.12 * franchise_boost
        _debug_filter_step(
            "HYBRID",
            "anchor genre/franchise",
            before,
            _count_positive(final),
            note="genre_overlap>2 or franchise>=0.75 (+franchise boost)",
        )

    # Keep existing broader genre alignment logic.
    if effective_genres and anchor_idx is not None:
        g_overlap = np.array([
            len(effective_genres & (set(g) if isinstance(g, list) else set()))
            for g in df['genres']
        ])
        genre_ok = (g_overlap >= 1)
        genre_filtered = final * genre_ok
        if int(genre_filtered.astype(bool).sum()) >= top_n * 2:
            _debug_filter_step(
                "HYBRID",
                "effective genre gate",
                _count_positive(final),
                int(genre_filtered.astype(bool).sum()),
                note="genre_overlap>=1",
            )
            final = genre_filtered

    # ── Pick top_n candidates ─────────────────────────────────────────────────
    fetch_n  = top_n * 3 if diversify else top_n * 2
    top_idxs = np.argsort(final)[::-1][:fetch_n]
    top_idxs = top_idxs[final[top_idxs] > 0.0]
    print(
        f"[DEBUG][HYBRID] ranked pool size={len(top_idxs)} "
        f"(fetch_n={fetch_n}, diversify={diversify})"
    )

    # ── MMR diversification ───────────────────────────────────────────────────
    if diversify and _sbert_vecs is not None and len(top_idxs) > top_n:
        selected = _mmr(top_idxs, final, top_n)
        print(f"[DEBUG][HYBRID] MMR selected {len(selected)} movies from ranked pool")
    else:
        selected = top_idxs[:top_n]
        print(f"[DEBUG][HYBRID] top-{len(selected)} selected directly by final score")

    # ── Build result objects ──────────────────────────────────────────────────
    results = []
    for rank, idx in enumerate(selected, start=1):
        if idx >= len(df):
            continue
        row   = df.iloc[idx]
        score = float(final[idx])
        fm    = float(fame_norm[idx])

        cand_genres = set(row.get('genres', []) if isinstance(row.get('genres', []), list) else [])
        cand_keywords = _meaningful_keywords(row.get('keywords', []))
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
        vote_confidence_val = float(vote_confidence[idx]) if idx < len(vote_confidence) else 0.0
        rating_norm_val = float(rating_norm[idx]) if idx < len(rating_norm) else 0.0
        franchise_val = float(franchise_boost[idx]) if idx < len(franchise_boost) else 0.0

        if genre_only_mode:
            components = [
                ("vote_confidence", vote_confidence_val, 0.55),
                ("fame", fm, 0.30),
                ("rating_norm", rating_norm_val, 0.15),
            ]
        else:
            components = [
                ("sim_norm", sim_norm_val, 0.70),
                ("vote_confidence", vote_confidence_val, 0.17),
                ("fame", fm, 0.10),
                ("rating_norm", rating_norm_val, 0.03),
            ]

        top_reasons = sorted(components, key=lambda x: x[1] * x[2], reverse=True)[:3]

        print(f"\n[DEBUG][HYBRID][RANK {rank:02d}] {row.get('title', '')} (idx={idx})")
        print(
            f"  final_score={score:.4f} | vote_avg={float(row.get('vote_average', 0.0)):.2f} | "
            f"vote_count={int(row.get('vote_count', 0))}"
        )
        print(
            f"  base signals: tfidf={tfidf_val:.4f}, sbert={sbert_val:.4f}, cf={cf_val:.4f}, "
            f"genre_bonus={genre_bonus_val:.4f}, total_sim={total_sim_val:.4f}"
        )
        print(
            f"  fused signals: sim_norm={sim_norm_val:.4f}, vote_confidence={vote_confidence_val:.4f}, "
            f"fame={fm:.4f}, rating_norm={rating_norm_val:.4f}, franchise_boost={franchise_val:.4f}"
        )
        _debug_weight_breakdown("HYBRID", components)
        print(
            "  why recommended: "
            + ", ".join(f"{name}={raw * wt:.4f}" for name, raw, wt in top_reasons)
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

    # Query genre set from chips (used as structural filtering only).
    query_genres = {c for c in (selected_chips or []) if c in SUPPORTED_GENRES}

    # Enforce minimum vote_average floor. If caller sends lower value, keep 5.0.
    min_vote_avg = max(5.0, float(min_rating))

    # If user selects only genre tags (no movie title, no free text), rank mostly
    # by vote_count and vote_average within matching genres.
    genre_only_mode = bool(
        anchor_idx is None and
        not (free_text or '').strip() and
        bool(selected_chips) and
        all(c in SUPPORTED_GENRES for c in (selected_chips or []))
    )

    effective_decade_filter = list(decade_filter or [])
    if anchor_idx is not None:
        anchor_year = int(_df.loc[anchor_idx].get('release_year', 2000))
        effective_decade_filter = _ensure_anchor_decade(effective_decade_filter, anchor_year)

    # ── Routing: title-only uses semantic recommender ────────────────────────
    title_only_mode = bool(anchor_idx is not None and not (free_text or '').strip() and not (selected_chips or []))

    _debug_stage_header("REQUEST", "Incoming recommendation request")
    print(f"[DEBUG][REQUEST] movie_title      : {movie_title.strip() if movie_title else '-'}")
    print(f"[DEBUG][REQUEST] free_text        : {_fmt_preview_text(free_text or '', max_chars=180)}")
    print(f"[DEBUG][REQUEST] selected_chips   : {_fmt_list(selected_chips or [], max_items=20)}")
    print(f"[DEBUG][REQUEST] query_text       : {_fmt_preview_text(query_text, max_chars=180)}")
    print(f"[DEBUG][REQUEST] query_genres     : {_fmt_list(sorted(list(query_genres)), max_items=20)}")
    print(f"[DEBUG][REQUEST] language_codes   : {_fmt_list(language_codes or [], max_items=20)}")
    print(f"[DEBUG][REQUEST] decade_filter    : {_fmt_list(effective_decade_filter or [], max_items=20)}")
    print(f"[DEBUG][REQUEST] top_n/min_rating : {top_n} / {min_vote_avg:.1f}")
    if anchor_idx is not None:
        print(f"[DEBUG][REQUEST] resolved_anchor  : idx={anchor_idx}, title={anchor_label}")
        _debug_movie_meta("Resolved anchor movie", _df.loc[anchor_idx], include_overview=True)
        if effective_decade_filter != (decade_filter or []):
            print(
                f"[DEBUG][REQUEST] decade filter auto-adjusted to include anchor decade "
                f"({_decade_label_from_year(int(_df.loc[anchor_idx].get('release_year', 2000)))})"
            )
    else:
        print("[DEBUG][REQUEST] resolved_anchor  : none (text/chip mode)")
    print(
        f"[DEBUG][REQUEST] route            : "
        f"{'semantic(title-only)' if title_only_mode else 'hybrid'}"
    )

    if title_only_mode:
        results = _semantic_recommend_from_anchor(
            anchor_idx      = anchor_idx,
            language_codes  = language_codes,
            decade_filter   = effective_decade_filter,
            top_n           = top_n,
            min_vote_avg    = min_vote_avg,
            year_window     = 12,
            min_genre_overlap = 1,
            min_vote_count  = 20,
        )

        # Fallback to hybrid if SBERT is unavailable or filters are too strict.
        if not results:
            print("[DEBUG][REQUEST] semantic returned empty; switching to hybrid fallback")
            results = _hybrid_recommend(
                anchor_idx     = anchor_idx,
                query_text     = query_text,
                query_genres   = query_genres,
                language_codes = language_codes,
                decade_filter  = effective_decade_filter,
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
            decade_filter  = effective_decade_filter,
            top_n          = top_n,
            min_vote_avg   = min_vote_avg,
            genre_only_mode = genre_only_mode,
            diversify      = diversify,
        )

    print(f"[DEBUG][REQUEST] completed with {len(results)} recommendations")

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
        # Popularity sort follows raw vote_count.
        return sorted(results, key=lambda m: m.vote_count, reverse=True)
    elif sort_by == "newest":
        return sorted(results, key=lambda m: m.year, reverse=True)
    else:  # best_match
        return sorted(results, key=lambda m: m.weighted_score, reverse=True)
