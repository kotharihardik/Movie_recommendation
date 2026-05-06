
from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

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
    weighted_score: float   # normalised [0,1] shown as match %
    fame_score:     float   # popularity signal
    justification:  str = ""
    justification_source: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_engine_ready:           bool = False
_df:                     Optional[pd.DataFrame] = None
_tfidf_matrix            = None
_tfidf:                  Optional[TfidfVectorizer] = None
_title_to_idx:           Optional[pd.Series] = None
_embed_model             = None          # sentence-transformer instance
_embed_vecs              = None          # (N, D) float32 numpy array, L2-normalised
_embed_model_name:       str = ""        # which model was loaded
_embed_needs_prefix:     bool = False
_reranker                = None          # cross-encoder reranker
_reranker_model_name:     str = ""
_knn                     = None
_movie_vecs              = None
_fame_scores:            Optional[np.ndarray] = None
_vote_confidence_scores: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_GENRES = {
    "Action", "Romance", "Thriller", "Drama", "Comedy",
    "Horror", "Family", "Historical", "Crime", "Sci-Fi",
}

# Genre combination → tone descriptor for the description prefix
_GENRE_TONE_MAP: dict[frozenset, str] = {
    frozenset(["Comedy", "Crime"]):                    "A chaotic comedy caper",
    frozenset(["Comedy", "Crime", "Family"]):          "A lighthearted family comedy caper",
    frozenset(["Action", "Romance", "Thriller"]):      "An intense action thriller with romantic elements",
    frozenset(["Action", "Thriller"]):                 "A high-stakes action thriller",
    frozenset(["Action", "Thriller", "Crime"]):        "A gritty crime action thriller",
    frozenset(["Action", "Adventure", "Thriller"]):    "A pulse-pounding action adventure thriller",
    frozenset(["Action", "Adventure", "Comedy"]):      "An adventurous action comedy",
    frozenset(["Action", "Romance"]):                  "A romantic action film",
    frozenset(["Action", "Crime"]):                    "A hard-hitting crime action film",
    frozenset(["Drama", "Romance"]):                   "An emotional romantic drama",
    frozenset(["Drama", "Crime", "Thriller"]):         "A dark psychological crime drama",
    frozenset(["Drama", "Thriller"]):                  "A gripping dramatic thriller",
    frozenset(["Drama"]):                              "A serious dramatic film",
    frozenset(["Crime", "Thriller", "Mystery"]):       "A dark psychological mystery thriller",
    frozenset(["Crime", "Thriller"]):                  "A tense crime thriller",
    frozenset(["Comedy", "Romance"]):                  "A lighthearted romantic comedy",
    frozenset(["Comedy", "Drama"]):                    "A bittersweet comedy drama",
    frozenset(["Comedy"]):                             "A lighthearted comedy",
    frozenset(["Horror"]):                             "A dark horror film",
    frozenset(["Horror", "Thriller"]):                 "A chilling horror thriller",
    frozenset(["Family", "Comedy"]):                   "A warm family comedy",
    frozenset(["Sci-Fi", "Action"]):                   "A futuristic sci-fi action film",
    frozenset(["Historical", "Drama"]):                "An epic historical drama",
    frozenset(["Historical", "Action"]):               "An epic historical action film",
    frozenset(["Romance"]):                            "A romantic film",
    frozenset(["Action"]):                             "An action film",
    frozenset(["Thriller"]):                           "A psychological thriller",
}

_GENRE_INTENT_MAP: dict[str, str] = {
    "Action": "An action-packed, high-energy film with stunts, chases, fights, and big set pieces.",
    "Romance": "A romantic love story with soulful music, songs, singers, musicians, melodic moments, emotional chemistry, heartfelt relationships, longing, heartbreak, and falling in love.",
    "Thriller": "A tense suspenseful thriller with danger, twists, mystery, tension, and gripping uncertainty.",
    "Drama": "A character-driven emotional drama with interpersonal conflict, life struggles, and layered relationships.",
    "Comedy": "A funny, light-hearted comedy with playful banter, situational humor, and charming characters.",
    "Horror": "A scary horror film with fear, darkness, eerie atmosphere, and terrifying moments.",
    "Family": "A warm family film with wholesome relationships, togetherness, and emotional bonds.",
    "Historical": "An epic period story set in the past with history, culture, and dramatic stakes.",
    "Crime": "A crime story with investigations, gangsters, heists, law, betrayal, and criminal conflict.",
    "Sci-Fi": "A futuristic science fiction story with advanced technology, space travel, robots, aliens, time travel, and discovery.",
}

_MOOD_INTENT_MAP: dict[str, str] = {
    "Revenge": "vengeful, dark, intense, and driven by retaliation",
    "Feel-Good": "uplifting, heartwarming, positive, cheerful, charming, and hopeful",
    "Tear-Jerker": "emotional, heartbreaking, moving, tragic, and bittersweet",
    "Dance": "musical, energetic, celebratory, and full of songs and performances",
    "Epic": "grand, sweeping, large-scale, and emotionally big",
    "Road Trip": "a journey, travel, self-discovery, adventure, and changing relationships",
    "Suspense": "mysterious, tense, twisty, and full of suspense",
    "Inspirational": "motivating, hopeful, uplifting, and inspiring",
}

_CHIP_AFFINITY_TERMS: dict[str, set[str]] = {
    "Action": {
        "action", "fight", "fights", "chase", "chases", "stunt", "stunts",
        "battle", "battles", "explosion", "explosions", "hero", "war",
    },
    "Romance": {
        "love", "lovers", "lover", "romance", "romantic", "relationship", "relationships",
        "marriage", "married", "couple", "couples", "heart", "heartbreak", "longing",
        "yearning", "music", "musical", "song", "songs", "singer", "singers",
        "musician", "musicians", "singing", "affair", "chemistry", "passion", "feelings",
        "soulmate", "crush", "courtship",
    },
    "Thriller": {
        "suspense", "suspenseful", "thriller", "mystery", "twist", "twists", "tension",
        "danger", "investigation", "secret", "gripping", "chase", "detective",
    },
    "Drama": {
        "emotion", "emotional", "family", "struggle", "struggles", "life", "tragedy",
        "relationship", "relationships", "conflict", "journey", "society", "heart",
    },
    "Comedy": {
        "funny", "hilarious", "humor", "humour", "comedy", "laugh", "laughs", "witty",
        "banter", "chaos", "misunderstanding", "comic",
    },
    "Horror": {
        "horror", "scary", "fear", "ghost", "ghosts", "haunted", "killer", "blood",
        "terrifying", "supernatural", "evil", "dark",
    },
    "Family": {
        "family", "children", "kids", "parents", "father", "mother", "brother", "sister",
        "home", "wholesome", "together", "bond",
    },
    "Historical": {
        "history", "historical", "period", "king", "queen", "empire", "war", "battle",
        "royal", "mughal", "freedom", "biopic",
    },
    "Crime": {
        "crime", "criminal", "gangster", "gang", "mafia", "heist", "police",
        "investigation", "murder", "law", "detective", "corruption",
    },
    "Sci-Fi": {
        "science fiction", "sci fi", "sci-fi", "future", "futuristic", "space", "spaceship",
        "planet", "galaxy", "universe", "alien", "aliens", "robot", "robots", "android",
        "machine", "technology", "time travel", "quantum", "experiment", "scientist",
        "cyber", "digital", "virtual", "clone", "parallel",
    },
    "Revenge": {"revenge", "vengeance", "retaliation", "payback"},
    "Feel-Good": {"uplifting", "heartwarming", "hopeful", "cheerful", "positive", "feel good", "inspiring"},
    "Tear-Jerker": {"emotional", "heartbreaking", "sad", "tragic", "teary", "moving"},
    "Dance": {"dance", "dancing", "song", "songs", "music", "musical", "performance", "performances"},
    "Epic": {"epic", "grand", "sweeping", "scale", "battle", "journey"},
    "Road Trip": {"road trip", "journey", "travel", "adventure", "trip"},
    "Suspense": {"suspense", "tense", "twist", "mystery", "thriller"},
    "Inspirational": {"inspiring", "motivating", "hopeful", "overcome", "perseverance", "success"},
}

_GENERIC_KW = {
    "love", "romance", "action", "drama", "comedy", "thriller",
    "family", "friendship", "fight", "hero", "villain", "movie",
    "film", "story", "life", "man", "woman", "girl", "boy",
}

_PLOT_STOPWORDS = set(ENGLISH_STOP_WORDS) | _GENERIC_KW | {
    "people", "person", "thing", "things", "day", "days", "year", "years",
    "new", "old", "young", "big", "small", "world", "city", "town", "village",
    "group", "help", "find", "finds", "gets", "get", "go", "goes", "come", "comes",
    "take", "takes", "want", "wants", "must", "set", "based", "around", "later",
    "one", "two", "three", "first", "last",
}

# Single fixed embedding model (fast startup, no heavy fallback downloads)
_CANDIDATE_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", False),
]

# Scoring weights — semantic mode (title-only)
SEMANTIC_WEIGHTS = {
    "anchor_sim_rank":    0.10,   # blended TF-IDF+embed+CF percentile rank
    "embed_rank":         0.08,   # direct semantic/tone similarity (embedding-only)
    "cross_encoder_rank": 0.24,   # direct query-document relevance reranker
    "plot_jaccard":       0.20,   # plot-motif overlap from overviews
    "cast_jaccard":       0.08,
    "genre_jaccard":      0.06,
    "keyword_jaccard":    0.05,   # thematic keyword overlap
    "temporal_soft":      0.05,
    "vote_confidence":    0.03,
    "fame_score":         0.02,
    "director_match":     0.08,
    "franchise_boost":    0.02,
    "rating_norm":        0.00,
}

# Anchor similarity blend weights (TF-IDF, embed, CF)
ANCHOR_WEIGHTS = {"tfidf": 0.35, "embed": 0.60, "cf": 0.05}

# Cast-overlap Jaccard threshold that bypasses genre gate (franchise/sequel immunity)
CAST_IMMUNITY_JACCARD = 0.40

# Raw embed cosine threshold for keyword/theme gate
EMBED_THEME_GATE = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_list(values, max_items: int = 8) -> str:
    if not isinstance(values, list) or not values:
        return "-"
    shown = values[:max_items]
    extra = "" if len(values) <= max_items else f" ... (+{len(values) - max_items} more)"
    return ", ".join(str(v) for v in shown) + extra


def _fmt_text(text: str, max_chars: int = 220) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return (t[:max_chars].rstrip() + "...") if len(t) > max_chars else (t or "-")


def _dbg_movie(label: str, row: pd.Series, full: bool = False) -> None:
    print(f"\n[DEBUG] {label}")
    print(f"  title    : {row.get('title', '')}")
    print(f"  year/lang: {row.get('release_year', 0)} / {row.get('language', '')}")
    print(f"  vote     : avg={float(row.get('vote_average', 0)):.2f}  count={int(row.get('vote_count', 0))}")
    print(f"  genres   : {_fmt_list(row.get('genres', []), 10)}")
    print(f"  keywords : {_fmt_list(row.get('keywords', []), 12)}")
    print(f"  cast     : {_fmt_list(row.get('cast', []), 8)}")
    if full:
        print(f"  overview : {_fmt_text(row.get('overview', ''), 220)}")


def _dbg_header(scope: str, title: str) -> None:
    print(f"\n[DEBUG][{scope}] {'=' * 62}")
    print(f"[DEBUG][{scope}] {title}")


def _dbg_filter(scope: str, step: str, before: int, after: int, note: str = "") -> None:
    removed = max(0, before - after)
    pct = (100.0 * after / before) if before > 0 else 0.0
    sfx = f" | {note}" if note else ""
    print(f"[DEBUG][{scope}][FILTER] {step:<30} {before:5d} -> {after:5d}  (removed={removed:4d}, kept={pct:5.1f}%){sfx}")


def _dbg_weights(scope: str, components: list[tuple[str, float, float]]) -> None:
    print(f"[DEBUG][{scope}] score breakdown")
    for i, (name, raw, w) in enumerate(components, 1):
        print(f"  {i:>2}. {name:<24} raw={raw:.4f}  x  w={w:.2f}  =>  {w*raw:.4f}")


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Text / token helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOP = {
    "a","an","the","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","on","at","by",
    "for","with","about","as","into","through","his","her","their","its",
    "he","she","they","we","it","this","that","these","those","who","which",
    "when","where","how","what","not","no","nor","so","yet","both","either",
    "from","up","out","if","then","than","too","very","just","also",
}


def _clean_token(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()


def _tokenise(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return " ".join(t for t in tokens if t not in _STOP)


def _normalise_terms(values) -> set:
    if not isinstance(values, list):
        return set()
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _meaningful_kw(values) -> set:
    terms = _normalise_terms(values)
    return {t for t in terms if len(t.replace(" ", "")) >= 4 and t not in _GENERIC_KW}


def _plot_terms(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z]{3,}", str(text or "").lower())
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in _PLOT_STOPWORDS
    }


def _plot_terms_ordered(text: str, max_terms: int = 12) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z]{3,}", str(text or "").lower()):
        if len(token) < 4 or token in _PLOT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= max_terms:
            break
    return ordered


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _is_clean_title(x: str) -> bool:
    t = str(x).strip()
    if len(t) < 2:
        return False
    if re.fullmatch(r"[0-9]+", t):
        return False
    if re.fullmatch(r"[^a-zA-Z0-9]+", t):
        return False
    return True


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Convert raw scores to [0,1] percentile ranks for better spread."""
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0 or float(arr.max() - arr.min()) < 1e-12:
        return np.zeros(arr.size, dtype=float)
    return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Description builder  (the heart of semantic quality)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_tone(genres: list) -> str:
    """Map genre list to a tone descriptor using the lookup table."""
    if not genres:
        return "A film"
    genre_set = frozenset(str(g).strip() for g in genres if str(g).strip())

    # Exact match first
    if genre_set in _GENRE_TONE_MAP:
        return _GENRE_TONE_MAP[genre_set]

    # Best subset match (largest matching subset wins)
    best_match, best_size = "A film", 0
    for key, tone in _GENRE_TONE_MAP.items():
        overlap = len(key & genre_set)
        if overlap > best_size and key.issubset(genre_set):
            best_match, best_size = tone, overlap

    if best_size > 0:
        return best_match

    # Fallback: compose from raw genre names
    genre_str = ", ".join(str(g) for g in genres[:3])
    return f"A {genre_str} film"


def _build_chip_intent_text(selected_chips: list) -> str:
    """Turn selected genre/mood chips into a natural-language query intent."""
    genres = [chip for chip in (selected_chips or []) if chip in SUPPORTED_GENRES]
    moods = [chip for chip in (selected_chips or []) if chip not in SUPPORTED_GENRES]

    parts = []
    if genres:
        tone = _derive_tone(genres)
        genre_details = [
            _GENRE_INTENT_MAP.get(genre, f"A {genre.lower()} film.")
            for genre in genres
        ]
        parts.append(f"{tone}. {' '.join(genre_details)}")

    if moods:
        mood_details = [
            _MOOD_INTENT_MAP.get(mood, mood.lower())
            for mood in moods
        ]
        parts.append(f"The mood should feel {', '.join(mood_details)}.")

    return " ".join(parts).strip()


def _movie_text_blob(row: pd.Series) -> str:
    parts = [str(row.get("title", "")), str(row.get("overview", "")), str(row.get("tagline", ""))]
    keywords = row.get("keywords", []) or []
    if isinstance(keywords, list) and keywords:
        parts.append(" ".join(str(keyword) for keyword in keywords[:12]))
    return " ".join(parts).lower()


def _chip_affinity_terms(selected_chips: list) -> set[str]:
    terms: set[str] = set()
    for chip in selected_chips or []:
        terms |= _CHIP_AFFINITY_TERMS.get(chip, set())
    return terms


def _chip_affinity_arr(df: pd.DataFrame, selected_chips: list) -> np.ndarray:
    terms = _chip_affinity_terms(selected_chips)
    if not terms:
        return np.zeros(len(df), dtype=float)
    return np.array([
        sum(term in _movie_text_blob(row) for term in terms)
        for _, row in df.iterrows()
    ], dtype=float)


def _make_description(row: pd.Series) -> str:
    """Build the semantic description for the embedding model.

    Data checks showed that a simpler plot-first representation works better
    than a cast/genre-heavy prompt for Bollywood title-to-title matching.
    We therefore use only the title and overview here.
    """
    title = re.sub(r"\s+", " ", str(row.get("title", "") or "")).strip()
    overview = re.sub(r"\s+", " ", str(row.get("overview", "") or "")).strip()
    if title and overview:
        return f"{title}. {overview}"
    return title or overview


def _make_reranker_text(row: pd.Series) -> str:
    """Build a structured text profile for the cross-encoder reranker.

    The reranker needs more than a raw title+overview pair to separate movies
    that share only generic comedy/drama language. Anchoring the prompt with
    tone, motifs, cast, and director gives the model clearer similarity cues.
    """
    title = re.sub(r"\s+", " ", str(row.get("title", "") or "")).strip()
    tone = _derive_tone(row.get("genres", []))
    overview = _fmt_text(row.get("overview", ""), 220)
    motifs = " ".join(_plot_terms_ordered(row.get("overview", ""), 12))
    genres = ", ".join(str(g) for g in (row.get("genres", []) or [])[:4])
    cast = ", ".join(str(a) for a in (row.get("cast", []) or [])[:4])
    director = re.sub(r"\s+", " ", str(row.get("director", "") or "")).strip()

    parts = [f"{tone}."]
    if title:
        parts.append(f"Title: {title}.")
    if motifs:
        parts.append(f"Motifs: {motifs}.")
    if genres:
        parts.append(f"Genres: {genres}.")
    if cast:
        parts.append(f"Cast: {cast}.")
    if director and director != "Unknown":
        parts.append(f"Director: {director}.")
    if overview:
        parts.append(f"Overview: {overview}.")
    return " ".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF soup builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_soup(row: pd.Series) -> str:
    """Weighted TF-IDF soup with plot-first weighting.

    Title and overview are the primary relevance signals. Keywords are a
    secondary semantic hint. Genres/cast/director are supporting metadata,
    not the main relevance driver.
    """
    title_tok = _tokenise(str(row.get("title", "") or ""))
    kw_tok  = " ".join(_clean_token(k) for k in (row["keywords"] or []))
    g_tok   = " ".join(_clean_token(g) for g in (row["genres"]   or []))
    c_tok   = " ".join(_clean_token(a) for a in (row["cast"]     or []))
    d_tok   = _clean_token(row.get("director", "") or "")
    ov_tok  = _tokenise(str(row.get("overview", "") or ""))
    return f"{title_tok} {title_tok} {kw_tok} {kw_tok} {g_tok} {c_tok} {d_tok} {ov_tok} {ov_tok} {ov_tok} {ov_tok}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Popularity / reliability priors
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fame_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Star-power fame heuristic.

    Rules:
    - Only count appearances in movies with vote_count > 50 (filters unknown films)
    - Billing position weighted: pos-1 = 1.0, pos-2 = 0.7, pos-3 = 0.5
    - Director contributes 0.45 × log(appearances)
    - Final score is MinMax-scaled to [0, 1]
    """
    from collections import Counter

    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).clip(lower=0)
    qualified = (vc > 50).values

    actor_counts: Counter = Counter()
    dir_counts:   Counter = Counter()

    for (_, row), is_q in zip(df.iterrows(), qualified):
        if not bool(is_q):
            continue
        d = str(row.get("director", "") or "")
        if d:
            dir_counts[d] += 1
        for actor in (row.get("cast", []) or [])[:5]:
            actor_counts[actor] += 1

    pos_weights = [1.0, 0.7, 0.5]
    raw = np.zeros(len(df), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        cast_list = (row.get("cast", []) or [])[:3]
        cast_term = sum(
            pos_weights[p] * math.log1p(actor_counts.get(a, 0))
            for p, a in enumerate(cast_list)
        )
        d = str(row.get("director", "") or "")
        dir_term = 0.45 * math.log1p(dir_counts.get(d, 0))
        raw[i] = cast_term + dir_term

    scaler = MinMaxScaler()
    return scaler.fit_transform(raw.reshape(-1, 1)).flatten()


def _compute_vote_confidence(df: pd.DataFrame) -> np.ndarray:
    """Log-scaled vote_count normalised to [0,1]. Measures result reliability."""
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).clip(lower=0)
    vc_max = float(vc.max()) if len(vc) else 0.0
    if vc_max <= 0:
        return np.zeros(len(df), dtype=float)
    return (np.log1p(vc) / np.log1p(vc_max + 1e-9)).to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Engine builder
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(df: pd.DataFrame) -> None:
    """Build all models from the cleaned dataframe. Called once at startup."""
    global _engine_ready, _df
    global _tfidf_matrix, _tfidf, _title_to_idx
    global _embed_model, _embed_vecs, _embed_model_name, _embed_needs_prefix
    global _knn, _movie_vecs
    global _fame_scores, _vote_confidence_scores

    if _engine_ready:
        return

    print("Building CineMatch recommendation engine…")
    t0 = time.time()

    _df = df.reset_index(drop=True).copy()

    # Weighted rating (Bayesian average)
    C = pd.to_numeric(_df["vote_average"], errors="coerce").fillna(0).mean()
    m = pd.to_numeric(_df["vote_count"],   errors="coerce").fillna(0).quantile(0.60)
    vc = pd.to_numeric(_df["vote_count"],  errors="coerce").fillna(0)
    va = pd.to_numeric(_df["vote_average"],errors="coerce").fillna(0)
    _df["weighted_rating"] = (vc / (vc + m)) * va + (m / (vc + m)) * C

    # ── 1. TF-IDF ──────────────────────────────────────────────────────────
    print("  [1/4] Building TF-IDF soup…")
    _df["soup"]        = _df.apply(_make_soup, axis=1)
    _df["description"] = _df.apply(_make_description, axis=1)

    _tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_features=50_000, sublinear_tf=True,
    )
    _tfidf_matrix = _tfidf.fit_transform(_df["soup"])

    def _normalize_title_key(value: str) -> str:
        text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"([aeiou])\1+", r"\1", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    _title_to_idx = pd.Series(_df.index, index=_df["title"].apply(_normalize_title_key))
    print(f"     TF-IDF matrix: {_tfidf_matrix.shape}")

    # ── 2. Semantic embedding model ────────────────────────────────────────
    print("  [2/4] Loading semantic embedding model…")
    _embed_model = None
    _embed_vecs  = None

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name, needs_prefix in _CANDIDATE_MODELS:
            try:
                print(f"     Trying: {model_name} …")
                _embed_model = SentenceTransformer(model_name, device=device)
                _embed_model_name    = model_name
                _embed_needs_prefix  = needs_prefix
                print(f"     Loaded: {model_name}  (prefix={'passage:' if needs_prefix else 'none'})")
                break
            except Exception as e:
                print(f"     {model_name} failed: {e}")
                _embed_model = None

        if _embed_model is not None:
            descriptions = _df["description"].tolist()
            if _embed_needs_prefix:
                descriptions = [f"passage: {d}" for d in descriptions]

            _embed_vecs = _embed_model.encode(
                descriptions,
                batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            print(f"     Embedding matrix: {_embed_vecs.shape}")
        else:
            print("     All embedding models failed - semantic scoring disabled.")

    except ImportError as e:
        print(f"     sentence-transformers not available ({e}); semantic scoring disabled.")

    # ── 3. SVD + KNN collaborative model ──────────────────────────────────
    print("  [3/4] Building SVD+KNN collaborative model…")
    try:
        cv = CountVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2, max_features=15_000)
        kw_text = _df.apply(
            lambda r: " ".join(
                [_clean_token(k) for k in (r["keywords"] or [])] +
                [_clean_token(g) for g in (r["genres"]   or [])] +
                [_clean_token(a) for a in (r["cast"]     or [])]
            ), axis=1,
        )
        kw_matrix = cv.fit_transform(kw_text)
        n_comp    = min(300, kw_matrix.shape[1] - 1)
        svd       = TruncatedSVD(n_components=n_comp, random_state=42)
        _movie_vecs = svd.fit_transform(kw_matrix)
        _knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=100)
        _knn.fit(_movie_vecs)
        print(f"     SVD: {_movie_vecs.shape}  KNN: ready")
    except Exception as e:
        print(f"       SVD/KNN failed ({e})")
        _knn = None
        _movie_vecs = None

    # ── 4. Popularity priors ───────────────────────────────────────────────
    print("  [4/4] Computing fame and vote-confidence scores…")
    _fame_scores            = _compute_fame_scores(_df)
    _vote_confidence_scores = _compute_vote_confidence(_df)
    print(
        f"     Fame:            [{_fame_scores.min():.3f}, {_fame_scores.max():.3f}]\n"
        f"     Vote-confidence: [{_vote_confidence_scores.min():.3f}, {_vote_confidence_scores.max():.3f}]"
    )

    _engine_ready = True
    print(f"Engine ready in {time.time() - t0:.1f}s  |  model={_embed_model_name or 'NONE'}")


# ─────────────────────────────────────────────────────────────────────────────
# Movie lookup
# ─────────────────────────────────────────────────────────────────────────────

def _find_movie_idx(query: str, language: str = None, exact_only: bool = False) -> Optional[int]:
    """Title lookup: exact first, then optional startswith/contains fallback."""
    if _title_to_idx is None or _df is None:
        return None

    def _normalize_title_key(value: str) -> str:
        text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"([aeiou])\1+", r"\1", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    q = _normalize_title_key(query)

    def pick(series_or_int):
        idxs = list(series_or_int.values) if isinstance(series_or_int, pd.Series) else [int(series_or_int)]
        if language:
            for i in idxs:
                if _df.loc[i, "language"] == language:
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
    if exact_only:
        return None
    return None


_SCIFI_GENRE_ALIASES = {"Sci-Fi"}
_GENRE_CANONICAL_ALIASES = {
    "sci fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "history": "Historical",
}


def _normalise_genre_label(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _canonical_genre_label(value) -> str:
    token = _normalise_genre_label(value)
    if not token:
        return ""
    return _GENRE_CANONICAL_ALIASES.get(token, str(value).strip())


def _genre_tokens(genres_val) -> set[str]:
    if isinstance(genres_val, list):
        raw_items = genres_val
    elif isinstance(genres_val, str):
        raw_items = re.split(r"[|,;/]", genres_val)
    else:
        return set()
    return {
        token
        for token in (_canonical_genre_label(item) for item in raw_items)
        if token
    }


def _has_genre_alias(genres_val, aliases: set[str]) -> bool:
    return bool(_genre_tokens(genres_val) & aliases)


def _is_scifi_genre(genres_val) -> bool:
    return _has_genre_alias(genres_val, _SCIFI_GENRE_ALIASES)


def _scifi_fame_fallback(top_n: int, min_vote_avg: float, language_codes: list, decade_filter: list) -> list[RecommendedMovie]:
    if _df is None or len(_df) == 0:
        return []

    pool = _df.copy()
    pool = pool[pool["title"].apply(_is_clean_title)]

    if language_codes:
        pool = pool[pool["language"].isin(language_codes)]

    pool = pool[_decade_mask(pool, decade_filter)]
    pool = pool[pd.to_numeric(pool["vote_average"], errors="coerce").fillna(0) >= min_vote_avg]
    pool = pool[pd.to_numeric(pool["vote_count"], errors="coerce").fillna(0) > 0]
    pool = pool[pool["genres"].apply(_is_scifi_genre)]

    if len(pool) == 0:
        return []

    _attach_priors(pool, _df)
    pool["semantic_score"] = pool["fame_score"]
    ranked = pool.sort_values(["fame_score", "vote_count", "vote_average"], ascending=False).head(top_n)
    return _build_results_semantic(ranked, set(), set(), set(), set())


# ─────────────────────────────────────────────────────────────────────────────
# Individual scorers
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_scores(anchor_idx: int) -> np.ndarray:
    if _tfidf_matrix is None:
        return np.zeros(len(_df))
    sims = linear_kernel(_tfidf_matrix[anchor_idx], _tfidf_matrix).flatten()
    sims[anchor_idx] = 0.0
    return sims


def _embed_scores_from_idx(anchor_idx: int) -> np.ndarray:
    """Cosine similarity from anchor movie's stored embedding."""
    if _embed_vecs is None:
        return np.zeros(len(_df))
    sims = (_embed_vecs @ _embed_vecs[anchor_idx]).flatten()
    sims[anchor_idx] = 0.0
    return sims


def _embed_scores_from_text(query_text: str) -> np.ndarray:
    """Encode free-text query and compute cosine similarity against all movies."""
    if _embed_vecs is None or _embed_model is None:
        return np.zeros(len(_df))
    # Use query prefix for e5; other models use plain text
    prefix = "query: " if _embed_needs_prefix else ""
    q_vec = _embed_model.encode(
        [f"{prefix}{query_text}"], normalize_embeddings=True
    )[0]
    return (_embed_vecs @ q_vec).flatten()


def _tfidf_scores_from_text(query_text: str) -> np.ndarray:
    """Compute TF-IDF similarity for free-text query."""
    if _tfidf is None or _tfidf_matrix is None:
        return np.zeros(len(_df))
    q_tfidf = _tfidf.transform([_clean_token(query_text)])
    sims = (linear_kernel(q_tfidf, _tfidf_matrix).flatten())
    return sims


def _knn_scores(anchor_idx: int) -> np.ndarray:
    out = np.zeros(len(_df))
    if _knn is None or _movie_vecs is None:
        return out
    dists, nn_idx = _knn.kneighbors(_movie_vecs[anchor_idx].reshape(1, -1), n_neighbors=100)
    for dist, idx in zip(dists.flatten(), nn_idx.flatten()):
        if idx != anchor_idx and idx < len(out):
            out[idx] = max(out[idx], 1.0 - dist)
    return out


def _anchor_bundle(anchor_idx: int, query_text: str = "") -> dict[str, np.ndarray]:
    """Compute all three anchor signals and blend them."""
    n = len(_df) if _df is not None else 0
    z = np.zeros(n, dtype=float)

    s_tfidf = _tfidf_scores(anchor_idx)
    s_embed = _embed_scores_from_idx(anchor_idx) if _embed_vecs is not None else z.copy()

    if query_text:
        s_text = _embed_scores_from_text(query_text)
        s_embed = 0.80 * s_embed + 0.20 * s_text

    s_cf  = _knn_scores(anchor_idx)
    total = (
        ANCHOR_WEIGHTS["tfidf"] * s_tfidf +
        ANCHOR_WEIGHTS["embed"] * s_embed +
        ANCHOR_WEIGHTS["cf"]    * s_cf
    )
    total[anchor_idx] = 0.0
    return {"tfidf": s_tfidf, "embed": s_embed, "cf": s_cf, "total": total}


def _ensure_reranker() -> None:
    """Lazy-load the cross-encoder reranker used for title-only semantic queries."""
    global _reranker, _reranker_model_name

    if _reranker is not None:
        return

    try:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/stsb-roberta-base"
        print(f"     Trying reranker: {model_name} …")
        _reranker = CrossEncoder(model_name, device="cpu", tokenizer_args={"use_fast": False})
        _reranker_model_name = model_name
        print(f"     Loaded reranker: {model_name}")
    except Exception as e:
        print(f"     reranker unavailable: {e}")
        _reranker = None
        _reranker_model_name = ""


# ─────────────────────────────────────────────────────────────────────────────
# Franchise / series detection
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_franchise(title: str) -> str:
    t = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii").lower()
    t = re.split(r"\s[-–—]\s|:", t, maxsplit=1)[0].strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\b(?:the|a|an)$", "", t).strip()
    t = re.sub(r"\b(?:part|chapter|episode)\b\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+|[ivx]+)?$", "", t).strip()
    t = re.sub(r"\b(?:beginning|conclusion|epic|returns?|again|rise|rises|return|next level)\b$", "", t).strip()
    t = re.sub(r"\b(?:[0-9]+|[ivx]+)\b$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _franchise_scores(df: pd.DataFrame, anchor_title: str) -> np.ndarray:
    anchor_base   = _normalise_franchise(anchor_title)
    anchor_tokens = set(anchor_base.split()) if anchor_base else set()
    scores = np.zeros(len(df), dtype=float)

    for i, t in enumerate(df["title"].fillna("").astype(str).tolist()):
        cand_base   = _normalise_franchise(t)
        cand_tokens = set(cand_base.split()) if cand_base else set()
        if not cand_base:
            continue
        if cand_base == anchor_base:
            scores[i] = 1.0
        elif anchor_tokens and anchor_tokens.issubset(cand_tokens):
            scores[i] = 1.0
        elif cand_tokens and cand_tokens.issubset(anchor_tokens):
            scores[i] = 0.95
        else:
            scores[i] = _jaccard(anchor_tokens, cand_tokens)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Decade / year helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decade_label(year: int) -> str:
    if year >= 2020: return "2020s"
    if year >= 2010: return "2010s"
    if year >= 2000: return "2000s"
    if year >= 1990: return "1990s"
    return "Classic (<1990)"


def _ensure_anchor_decade(decade_filter: list, anchor_year: int, include_old_movies: bool = False) -> list:
    if not decade_filter:
        return decade_filter
    
    # If in Classic mode, we only want to stay in the classic decades.
    # If the anchor is modern, we don't want to bring modern results into a classic search.
    if include_old_movies:
        return decade_filter

    updated = list(decade_filter)
    label = _decade_label(anchor_year)
    if label == "Classic (<1990)":
        if not any("Classic" in str(d) for d in updated):
            updated.append(label)
    elif label not in updated:
        updated.append(label)
    return updated


def _decade_mask(df: pd.DataFrame, decade_filter: list) -> np.ndarray:
    if not decade_filter:
        return np.ones(len(df), dtype=bool)
    masks = []
    for d in decade_filter:
        yr = df["release_year"]
        if d == "2020s":         masks.append((yr >= 2020).values)
        elif d == "2010s":       masks.append(((yr >= 2010) & (yr < 2020)).values)
        elif d == "2000s":       masks.append(((yr >= 2000) & (yr < 2010)).values)
        elif d == "1990s":       masks.append(((yr >= 1990) & (yr < 2000)).values)
        elif "Classic" in str(d):masks.append((yr < 1990).values)
    return np.any(np.stack(masks, axis=0), axis=0) if masks else np.ones(len(df), dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Genre overlap helper
# ─────────────────────────────────────────────────────────────────────────────

def _genre_overlap_arr(df: pd.DataFrame, query_genres: set) -> np.ndarray:
    if not query_genres:
        return np.zeros(len(df))
    return np.array([
        _jaccard(query_genres, _genre_tokens(g))
        for g in df["genres"]
    ])


def _chip_exact_fallback(
    selected_genres: set,
    selected_chips: list,
    language_codes: list,
    decade_filter: list,
    top_n: int,
    min_vote_avg: float,
) -> list[RecommendedMovie]:
    """Return a simple exact-tag fallback when the semantic ranking looks weak."""
    if _df is None or len(_df) == 0:
        return []

    pool = _df.copy()
    pool = pool[pool["title"].apply(_is_clean_title)]

    if language_codes:
        pool = pool[pool["language"].isin(language_codes)]

    pool = pool[_decade_mask(pool, decade_filter)]
    pool = pool[pd.to_numeric(pool["vote_average"], errors="coerce").fillna(0) >= min_vote_avg]
    pool = pool[pd.to_numeric(pool["vote_count"], errors="coerce").fillna(0) > 0]

    if selected_genres:
        pool = pool[pool["genres"].apply(lambda g: bool(_genre_tokens(g) & selected_genres))]

    if len(pool) == 0:
        return []

    intent_text = _build_chip_intent_text(selected_chips)
    if intent_text:
        pool["intent_raw"] = _embed_scores_from_text(intent_text)
        pool["intent_rank"] = _percentile_rank(pool["intent_raw"].to_numpy(dtype=float))
    else:
        pool["intent_rank"] = 0.0

    affinity_raw = _chip_affinity_arr(pool, selected_chips)
    pool["affinity_rank"] = _percentile_rank(affinity_raw)

    pool["genre_hit_count"] = pool["genres"].apply(lambda g: len(_genre_tokens(g) & selected_genres) if selected_genres else 0)
    pool["genre_purity"] = pool["genres"].apply(
        lambda g: (len(_genre_tokens(g) & selected_genres) / max(len(_genre_tokens(g)), 1)) if selected_genres else 0.0
    )

    _attach_priors(pool, _df)
    pool["fallback_score"] = (
        0.34 * pool["intent_rank"] +
        0.28 * pool["affinity_rank"] +
        0.18 * pool["genre_purity"] +
        0.08 * _percentile_rank(pool["genre_hit_count"].to_numpy(dtype=float)) +
        0.06 * pool["vote_confidence"] +
        0.04 * pool["fame_score"] +
        0.02 * pool["rating_norm"]
    )
    ranked = pool.sort_values(
        ["fallback_score", "genre_hit_count", "vote_confidence", "fame_score", "rating_norm"],
        ascending=False,
    ).head(top_n)
    ranked["semantic_score"] = ranked["fallback_score"]
    return _build_results_semantic(ranked, selected_genres, set(), set(), set())


# ─────────────────────────────────────────────────────────────────────────────
# Shared prior attachment
# ─────────────────────────────────────────────────────────────────────────────

def _attach_priors(frame: pd.DataFrame, base_df: pd.DataFrame) -> None:
    vc_all   = _vote_confidence_scores if _vote_confidence_scores is not None else np.zeros(len(base_df))
    fame_all = _fame_scores            if _fame_scores            is not None else np.zeros(len(base_df))
    frame["vote_confidence"] = pd.Series(vc_all,   index=base_df.index).reindex(frame.index).fillna(0.0)
    frame["fame_score"]      = pd.Series(fame_all, index=base_df.index).reindex(frame.index).fillna(0.0)
    frame["rating_norm"] = (
        pd.to_numeric(frame["weighted_rating"], errors="coerce")
        .fillna(0).clip(0, 10) / 10.0
    )


def _score_frame(frame: pd.DataFrame) -> pd.Series:
    return (
        SEMANTIC_WEIGHTS["anchor_sim_rank"]    * frame["anchor_sim_rank"]    +
        SEMANTIC_WEIGHTS["embed_rank"]         * frame["embed_rank"]         +
        SEMANTIC_WEIGHTS["cross_encoder_rank"] * frame["cross_encoder_rank"] +
        SEMANTIC_WEIGHTS["plot_jaccard"]       * frame["plot_jaccard"]       +
        SEMANTIC_WEIGHTS["cast_jaccard"]       * frame["cast_jaccard"]       +
        SEMANTIC_WEIGHTS["genre_jaccard"]      * frame["genre_jaccard"]      +
        SEMANTIC_WEIGHTS["keyword_jaccard"]    * frame["keyword_jaccard"]    +
        SEMANTIC_WEIGHTS["temporal_soft"]      * frame["temporal_soft"]      +
        SEMANTIC_WEIGHTS["vote_confidence"]    * frame["vote_confidence"]    +
        SEMANTIC_WEIGHTS["fame_score"]         * frame["fame_score"]         +
        SEMANTIC_WEIGHTS["director_match"]     * frame["director_match"]     +
        SEMANTIC_WEIGHTS["franchise_boost"]    * frame["franchise_boost"]    +
        SEMANTIC_WEIGHTS["rating_norm"]        * frame["rating_norm"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Semantic recommender  (title-only route)
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_recommend(
    anchor_idx:        int,
    language_codes:    list,
    decade_filter:     list,
    top_n:             int,
    min_vote_avg:      float = 5.0,
    year_window:       int   = 12,
    min_genre_overlap: int   = 1,
    min_vote_count:    int   = 20,
) -> list[RecommendedMovie]:

    if _df is None:
        return []

    df    = _df
    q_row = df.loc[anchor_idx]
    anchor_title_exact = str(q_row.get("title", "")).strip().lower()

    sims     = _anchor_bundle(anchor_idx)
    fran_sc  = _franchise_scores(df, str(q_row.get("title", "")))

    q_genres  = _genre_tokens(q_row.get("genres", []))
    q_kw      = _meaningful_kw(q_row.get("keywords", []))
    q_cast    = set((q_row.get("cast") or [])[:5])
    q_director = _clean_token(q_row.get("director", ""))
    q_year    = int(q_row.get("release_year", 2000))
    q_plot    = _plot_terms(q_row.get("overview", ""))

    _dbg_header("SEMANTIC", "Title-only recommendation")
    _dbg_movie("Anchor movie", q_row, full=True)

    # ── Build candidate frame ────────────────────────────────────────────────
    cand = df.copy()
    _dbg_filter("SEMANTIC", "start", len(df), len(cand))

    cand["tfidf_raw"]       = pd.Series(sims["tfidf"], index=df.index)
    cand["embed_raw"]       = pd.Series(sims["embed"], index=df.index)
    cand["cf_raw"]          = pd.Series(sims["cf"],    index=df.index)
    cand["anchor_sim_raw"]  = pd.Series(sims["total"], index=df.index)
    cand["franchise_boost"] = pd.Series(fran_sc,       index=df.index)

    # Clean titles
    before = len(cand)
    cand = cand[cand["title"].apply(_is_clean_title)]
    _dbg_filter("SEMANTIC", "clean title", before, len(cand))

    # Exclude identical title
    before = len(cand)
    cand = cand[cand["title"].fillna("").astype(str).str.strip().str.lower() != anchor_title_exact]
    _dbg_filter("SEMANTIC", "exclude same title", before, len(cand))

    # Structural features
    cand["genre_overlap"]   = cand["genres"].apply(lambda g: len(q_genres & _genre_tokens(g)) if isinstance(g, list) else 0)
    cand["genre_jaccard"]   = cand["genres"].apply(lambda g: _jaccard(q_genres, _genre_tokens(g)) if isinstance(g, list) else 0.0)
    cand["cast_jaccard"]    = cand["cast"].apply(  lambda c: _jaccard(q_cast,   set(c[:5])) if isinstance(c, list) else 0.0)
    cand["director_match"]  = cand["director"].apply(lambda d: 1.0 if q_director and _clean_token(d) == q_director else 0.0)

    # ── Genre gate (with cast immunity for sequels) ──────────────────────────
    req_overlap = min(min_genre_overlap, len(q_genres)) if q_genres else 0
    if q_genres and req_overlap > 0:
        before = len(cand)
        cand = cand[
            (cand["genre_overlap"] >= req_overlap) |
            (cand["cast_jaccard"]  >= CAST_IMMUNITY_JACCARD)
        ]
        _dbg_filter("SEMANTIC", "genre gate", before, len(cand),
                    note=f"overlap>={req_overlap} or cast_j>={CAST_IMMUNITY_JACCARD}")

    # Keyword features
    if "keywords" in cand.columns:
        cand["kw_overlap"]    = cand["keywords"].apply(lambda k: len(q_kw & _meaningful_kw(k)) if isinstance(k, list) else 0)
        cand["keyword_jaccard"] = cand["keywords"].apply(lambda k: _jaccard(q_kw, _meaningful_kw(k)) if isinstance(k, list) else 0.0)
    else:
        cand["kw_overlap"]    = 0
        cand["keyword_jaccard"] = 0.0

    # Theme relevance is now handled by the score itself.
    # Keeping this as a no-op preserves recall and lets ranking decide.
    _dbg_filter("SEMANTIC", "theme gate", len(cand), len(cand), note="not applied; score handles relevance")

    # Temporal soft score
    yr_diff = np.abs(cand["release_year"].astype(float) - float(q_year))
    sigma   = float(max(year_window, 1))
    cand["temporal_soft"] = np.exp(-(yr_diff ** 2) / (2.0 * sigma ** 2))

    # Percentile rank of blended similarity (key fix: separates scores that cluster tightly)
    cand["anchor_sim_rank"] = _percentile_rank(
        pd.to_numeric(cand["anchor_sim_raw"], errors="coerce").fillna(0.0).to_numpy()
    )
    cand["embed_rank"] = _percentile_rank(
        pd.to_numeric(cand["embed_raw"], errors="coerce").fillna(0.0).to_numpy()
    )

    # Hard filters
    if language_codes:
        before = len(cand)
        cand = cand[cand["language"].isin(language_codes)]
        _dbg_filter("SEMANTIC", "language", before, len(cand), note=str(language_codes))

    before = len(cand)
    cand = cand[_decade_mask(cand, decade_filter)]
    _dbg_filter("SEMANTIC", "decade", before, len(cand), note=str(decade_filter))

    before = len(cand)
    cand = cand[
        (pd.to_numeric(cand["vote_average"], errors="coerce").fillna(0) >= min_vote_avg) |
        (cand["franchise_boost"] >= 0.75)
    ]
    _dbg_filter("SEMANTIC", "vote_avg gate", before, len(cand), note=f">={min_vote_avg:.1f}")

    before = len(cand)
    cand = cand[pd.to_numeric(cand["vote_count"], errors="coerce").fillna(0) > 0]
    _dbg_filter("SEMANTIC", "nonzero votes", before, len(cand))

    before = len(cand)
    cand = cand[
        (pd.to_numeric(cand["vote_count"], errors="coerce").fillna(0) > min_vote_count) |
        (cand["franchise_boost"] >= 0.75)
    ]
    _dbg_filter("SEMANTIC", "vote_count floor", before, len(cand), note=f">{min_vote_count}")

    # Optional year window (applied only if enough candidates remain)
    in_window = cand[np.abs(pd.to_numeric(cand["release_year"], errors="coerce").fillna(0) - q_year) <= year_window]
    if len(in_window) >= top_n:
        _dbg_filter("SEMANTIC", "year window", len(cand), len(in_window), note=f"±{year_window}yr")
        cand = in_window

    # Relaxed fill if still short
    if len(cand) < top_n:
        cand = _semantic_relaxed_fill(
            cand, df, anchor_idx, anchor_title_exact, sims, fran_sc,
            q_genres, q_kw, q_cast, q_plot, q_director, q_year, year_window, req_overlap,
            language_codes, decade_filter, min_vote_avg, min_vote_count, top_n,
        )

    cand["plot_jaccard"] = cand["overview"].apply(lambda ov: _jaccard(q_plot, _plot_terms(ov)))

    _attach_priors(cand, df)
    cand["cross_encoder_raw"] = 0.0
    cand["cross_encoder_rank"] = 0.0

    _ensure_reranker()
    if _reranker is not None and len(cand) > 0:
        rerank_pool = min(len(cand), max(top_n * 10, 80), 250)
        rerank_seed = cand.sort_values(
            ["director_match", "cast_jaccard", "plot_jaccard", "genre_jaccard", "anchor_sim_rank", "embed_rank", "vote_confidence", "fame_score"],
            ascending=False,
        ).head(rerank_pool)
        query_text = _make_reranker_text(q_row)
        candidate_texts = [_make_reranker_text(cand.loc[idx]) for idx in rerank_seed.index]
        pair_inputs = [(query_text, candidate_text) for candidate_text in candidate_texts]
        rerank_scores = np.asarray(_reranker.predict(pair_inputs, batch_size=16), dtype=float)
        cand.loc[rerank_seed.index, "cross_encoder_raw"] = rerank_scores
        cand.loc[rerank_seed.index, "cross_encoder_rank"] = _percentile_rank(rerank_scores)

    cand["semantic_score"] = _score_frame(cand)
    if anchor_idx in cand.index:
        cand.loc[anchor_idx, ["semantic_score", "cross_encoder_rank", "cross_encoder_raw"]] = 0.0

    # Ensure dtypes
    for col in ["semantic_score","anchor_sim_rank","embed_rank","cross_encoder_rank","cross_encoder_raw",
            "plot_jaccard","genre_jaccard","cast_jaccard","keyword_jaccard","temporal_soft","vote_confidence",
            "fame_score","director_match","franchise_boost","rating_norm","tfidf_raw","embed_raw","cf_raw"]:
        cand[col] = pd.to_numeric(cand.get(col, 0.0), errors="coerce").fillna(0.0)

    # Final ranking: franchise slots first, then by score
    franchise_quota = min(3, top_n)
    franchise_df = (
        cand[cand["franchise_boost"] >= 0.90]
        .sort_values(["semantic_score","vote_confidence","fame_score"], ascending=False)
        .head(franchise_quota)
    )
    remaining_df = (
        cand.drop(index=franchise_df.index, errors="ignore")
        .sort_values(["semantic_score","vote_confidence","fame_score"], ascending=False)
    )
    top = pd.concat([franchise_df, remaining_df]).head(top_n)

    print(f"[DEBUG][SEMANTIC] shortlist: {len(top)}  (franchise_slots={len(franchise_df)})")

    return _build_results_semantic(top, q_genres, q_kw, q_cast, q_plot)


def _semantic_relaxed_fill(
    cand, df, anchor_idx, anchor_title_exact, sims, fran_sc,
    q_genres, q_kw, q_cast, q_plot, q_director, q_year, year_window, req_overlap,
    language_codes, decade_filter, min_vote_avg, min_vote_count, top_n,
) -> pd.DataFrame:
    """Widen filters to fill up to top_n candidates when strict filtering is too aggressive."""
    print(f"[DEBUG][SEMANTIC] only {len(cand)} candidates < top_n={top_n}; running relaxed fill")

    relaxed = df.copy()
    relaxed = relaxed[relaxed["title"].apply(_is_clean_title)]
    relaxed = relaxed[relaxed["title"].fillna("").astype(str).str.strip().str.lower() != anchor_title_exact]

    relaxed["tfidf_raw"]       = pd.Series(sims["tfidf"], index=df.index).reindex(relaxed.index).fillna(0.0)
    relaxed["embed_raw"]       = pd.Series(sims["embed"], index=df.index).reindex(relaxed.index).fillna(0.0)
    relaxed["cf_raw"]          = pd.Series(sims["cf"],    index=df.index).reindex(relaxed.index).fillna(0.0)
    relaxed["anchor_sim_raw"]  = pd.Series(sims["total"], index=df.index).reindex(relaxed.index).fillna(0.0)
    relaxed["franchise_boost"] = pd.Series(fran_sc,       index=df.index).reindex(relaxed.index).fillna(0.0)
    relaxed["genre_jaccard"]   = relaxed["genres"].apply(lambda g: _jaccard(q_genres, _genre_tokens(g)) if isinstance(g, list) else 0.0)
    relaxed["cast_jaccard"]    = relaxed["cast"].apply(  lambda c: _jaccard(q_cast,   set(c[:5])) if isinstance(c, list) else 0.0)
    relaxed["director_match"]  = relaxed["director"].apply(lambda d: 1.0 if q_director and _clean_token(d) == q_director else 0.0)
    relaxed["plot_jaccard"]    = relaxed["overview"].apply(lambda ov: _jaccard(q_plot, _plot_terms(ov)))

    if "keywords" in relaxed.columns:
        relaxed["kw_overlap"]     = relaxed["keywords"].apply(lambda k: len(q_kw & _meaningful_kw(k)) if isinstance(k, list) else 0)
        relaxed["keyword_jaccard"]= relaxed["keywords"].apply(lambda k: _jaccard(q_kw, _meaningful_kw(k)) if isinstance(k, list) else 0.0)
    else:
        relaxed["kw_overlap"]     = 0
        relaxed["keyword_jaccard"]= 0.0

    # No hard gate in relaxed fill either; relevance is handled downstream.

    yr_diff = np.abs(pd.to_numeric(relaxed["release_year"], errors="coerce").fillna(0) - q_year)
    relaxed["temporal_soft"]   = np.exp(-(yr_diff ** 2) / (2.0 * float(max(year_window, 1)) ** 2))
    relaxed["anchor_sim_rank"] = _percentile_rank(
        pd.to_numeric(relaxed["anchor_sim_raw"], errors="coerce").fillna(0.0).to_numpy()
    )
    relaxed["embed_rank"] = _percentile_rank(
        pd.to_numeric(relaxed["embed_raw"], errors="coerce").fillna(0.0).to_numpy()
    )
    relaxed["cross_encoder_raw"] = 0.0
    relaxed["cross_encoder_rank"] = 0.0
    _attach_priors(relaxed, df)
    relaxed["semantic_score"] = _score_frame(relaxed)
    if anchor_idx in relaxed.index:
        relaxed.loc[anchor_idx, "semantic_score"] = 0.0

    if language_codes:
        relaxed = relaxed[relaxed["language"].isin(language_codes)]
    relaxed = relaxed[_decade_mask(relaxed, decade_filter)]

    if q_genres and req_overlap > 0:
        relaxed = relaxed[
            (relaxed["genres"].apply(lambda g: len(q_genres & _genre_tokens(g)) if isinstance(g, list) else 0) >= req_overlap) |
            (relaxed["cast_jaccard"] >= CAST_IMMUNITY_JACCARD)
        ]

    relaxed = relaxed[pd.to_numeric(relaxed["vote_count"], errors="coerce").fillna(0) > 0]
    relaxed = relaxed[
        (pd.to_numeric(relaxed["vote_count"], errors="coerce").fillna(0) > min_vote_count) |
        (relaxed["franchise_boost"] >= 0.75)
    ]
    relaxed = relaxed[
        (pd.to_numeric(relaxed["vote_average"], errors="coerce").fillna(0) >= min_vote_avg) |
        (relaxed["franchise_boost"] >= 0.75)
    ]

    merged = pd.concat([cand, relaxed]).pipe(lambda f: f[~f.index.duplicated(keep="first")])
    _dbg_filter("SEMANTIC", "after relaxed merge", len(cand), len(merged))
    return merged


def _build_results_semantic(top: pd.DataFrame, q_genres: set, q_kw: set, q_cast: set, q_plot: set) -> list[RecommendedMovie]:
    fame_all = _fame_scores if _fame_scores is not None else np.zeros(len(_df))
    results = []

    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        cand_genres = _genre_tokens(row.get("genres", []))
        cand_kw     = _meaningful_kw(row.get("keywords", []))
        cand_cast   = set((row.get("cast", []) or [])[:5])

        tfidf_r  = _safe_float(row.get("tfidf_raw"))
        embed_r  = _safe_float(row.get("embed_raw"))
        cf_r     = _safe_float(row.get("cf_raw"))
        sim_rank = _safe_float(row.get("anchor_sim_rank"))
        emb_rank = _safe_float(row.get("embed_rank"))
        ce_rank  = _safe_float(row.get("cross_encoder_rank"))
        ce_raw   = _safe_float(row.get("cross_encoder_raw"))
        plot_j   = _safe_float(row.get("plot_jaccard"))
        genre_j  = _safe_float(row.get("genre_jaccard"))
        cast_j   = _safe_float(row.get("cast_jaccard"))
        kw_j     = _safe_float(row.get("keyword_jaccard"))
        kw_ov    = int(_safe_float(row.get("kw_overlap")))
        temporal = _safe_float(row.get("temporal_soft"))
        vc       = _safe_float(row.get("vote_confidence"))
        fame_s   = _safe_float(row.get("fame_score"))
        dir_m    = _safe_float(row.get("director_match"))
        fran     = _safe_float(row.get("franchise_boost"))
        rat_n    = _safe_float(row.get("rating_norm"))
        score    = _safe_float(row.get("semantic_score"))

        components = [
            ("anchor_sim_rank", sim_rank, SEMANTIC_WEIGHTS["anchor_sim_rank"]),
            ("embed_rank",      emb_rank, SEMANTIC_WEIGHTS["embed_rank"]),
            ("cross_encoder_rank", ce_rank, SEMANTIC_WEIGHTS["cross_encoder_rank"]),
            ("plot_jaccard",    plot_j,   SEMANTIC_WEIGHTS["plot_jaccard"]),
            ("cast_jaccard",    cast_j,   SEMANTIC_WEIGHTS["cast_jaccard"]),
            ("genre_jaccard",   genre_j,  SEMANTIC_WEIGHTS["genre_jaccard"]),
            ("keyword_jaccard", kw_j,     SEMANTIC_WEIGHTS["keyword_jaccard"]),
            ("temporal_soft",   temporal, SEMANTIC_WEIGHTS["temporal_soft"]),
            ("vote_confidence", vc,       SEMANTIC_WEIGHTS["vote_confidence"]),
            ("fame_score",      fame_s,   SEMANTIC_WEIGHTS["fame_score"]),
            ("director_match",  dir_m,    SEMANTIC_WEIGHTS["director_match"]),
            ("franchise_boost", fran,     SEMANTIC_WEIGHTS["franchise_boost"]),
            ("rating_norm",     rat_n,    SEMANTIC_WEIGHTS["rating_norm"]),
        ]
        top3 = sorted(components, key=lambda x: x[1] * x[2], reverse=True)[:3]

        print(f"\n[DEBUG][SEMANTIC][RANK {rank:02d}] {row.get('title','')} (idx={idx})")
        print(f"  score={score:.4f} | vote_avg={float(row.get('vote_average',0)):.2f} | vote_count={int(row.get('vote_count',0))}")
        print(f"  primitives: tfidf={tfidf_r:.4f}  embed={embed_r:.4f}  cf={cf_r:.4f}  sim_rank={sim_rank:.4f}  embed_rank={emb_rank:.4f}  ce_raw={ce_raw:.4f}  ce_rank={ce_rank:.4f}  plot_j={plot_j:.4f}  dir_match={dir_m:.4f}  kw_overlap={kw_ov}")
        _dbg_weights("SEMANTIC", components)
        print("  why: " + ", ".join(f"{n}={r*w:.4f}" for n, r, w in top3))
        print(f"  genres       : {_fmt_list(row.get('genres',[]),10)}")
        print(f"  keywords     : {_fmt_list(row.get('keywords',[]),12)}")
        print(f"  cast         : {_fmt_list(row.get('cast',[]),8)}")
        print(f"  plot_terms   : {_fmt_list(sorted(_plot_terms(row.get('overview', ''))),12)}")
        print(f"  shared_genres: {_fmt_list(sorted(q_genres & cand_genres),10)}")
        print(f"  shared_kw    : {_fmt_list(sorted(q_kw & cand_kw),12)}")
        print(f"  shared_cast  : {_fmt_list(sorted(q_cast & cand_cast),8)}")
        print(f"  shared_plot  : {_fmt_list(sorted(q_plot & _plot_terms(row.get('overview', ''))),12)}")

        fm = float(fame_all[idx]) if idx < len(fame_all) else 0.0
        results.append(RecommendedMovie(
            movie_id=str(row.get("id", idx)), title=str(row.get("title","")),
            original_title=str(row.get("original_title","")), language=str(row.get("language","")),
            year=int(row.get("release_year",0)), runtime=int(row.get("runtime",0)),
            vote_average=float(row.get("vote_average",0)), vote_count=int(row.get("vote_count",0)),
            genres=list(row.get("genres",[]) or []), director=str(row.get("director","")),
            cast=list(row.get("cast",[]) or []), poster_path=str(row.get("poster_path","")),
            tagline=str(row.get("tagline","")), overview=str(row.get("overview","")),
            budget=int(row.get("budget",0)), revenue=int(row.get("revenue",0)),
            weighted_score=float(max(0.0, min(1.0, score))), fame_score=fm,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid recommender  (text / chip / combined route)
# ─────────────────────────────────────────────────────────────────────────────

def _hybrid_recommend(
    anchor_idx:     Optional[int],
    query_text:     str,
    query_genres:   set,
    selected_chips: Optional[list] = None,
    language_codes: Optional[list] = None,
    decade_filter:  Optional[list] = None,
    top_n:          int = 10,
    min_vote_avg:   float = 5.0,
    genre_only_mode:bool  = False,
    include_old_movies: bool = False,
    diversify:      bool  = False,
    free_text_raw:  Optional[str] = None,
) -> list[RecommendedMovie]:

    df = _df
    n  = len(df)
    selected_chips = selected_chips or []
    language_codes = language_codes or []
    if decade_filter is None:
        decade_filter = ["2020s", "2010s", "2000s", "1990s", "Classic (<1990)"]

    _dbg_header("HYBRID", "Hybrid recommendation pipeline")

    fame_all = _fame_scores            if _fame_scores            is not None else np.zeros(n)
    vc_all   = _vote_confidence_scores if _vote_confidence_scores is not None else np.zeros(n)

    genre_bonus   = np.zeros(n)
    franchise_boost = np.zeros(n)
    q_genres_a = q_kw_a = q_cast_a = set()
    q_year     = None
    s_tfidf = s_embed = s_cf = total_sim = np.zeros(n)

    if anchor_idx is not None:
        q_row      = df.loc[anchor_idx]
        q_genres_a = set(q_row["genres"]) if isinstance(q_row.get("genres"), list) else set()
        q_kw_a     = _meaningful_kw(q_row.get("keywords", []))
        q_cast_a   = set((q_row.get("cast") or [])[:5])
        q_year     = int(q_row.get("release_year", 2000))
        franchise_boost = _franchise_scores(df, str(q_row.get("title", "")))

        bundle  = _anchor_bundle(anchor_idx, query_text)
        s_tfidf = bundle["tfidf"]
        s_embed = bundle["embed"]
        s_cf    = bundle["cf"]
        total_sim = bundle["total"]

        _dbg_movie("Anchor movie", q_row, full=True)
        print(f"[DEBUG][HYBRID] blend: tfidf×{ANCHOR_WEIGHTS['tfidf']} + embed×{ANCHOR_WEIGHTS['embed']} + cf×{ANCHOR_WEIGHTS['cf']}")
    else:
        s_embed   = _embed_scores_from_text(query_text) if query_text else np.zeros(n)
        s_tfidf   = _tfidf_scores_from_text(query_text) if query_text else np.zeros(n)
        total_sim = 0.65 * s_embed + 0.35 * s_tfidf
        print("[DEBUG][HYBRID] no anchor — text/chip mode blend: embed*0.65 + tfidf*0.35")
        print(f"  query: {_fmt_text(query_text)}")
        print(f"  genres: {sorted(query_genres)}")

    # Semantic gate for pure text queries (keep top-20% by raw embed score)
    text_gate = np.ones(n, dtype=bool)
    if anchor_idx is None and query_text.strip():
        cutoff = float(np.quantile(s_embed, 0.80)) if s_embed.any() else 0.0
        text_gate = s_embed >= cutoff
        if text_gate.sum() < top_n:
            forced = np.argsort(s_embed)[::-1][:max(top_n, 1)]
            text_gate = np.zeros(n, dtype=bool)
            text_gate[forced] = True
        print(f"[DEBUG][HYBRID] text gate: embed>={cutoff:.4f}  kept={text_gate.sum()}")

    # Genre overlap bonus
    effective_genres = query_genres | q_genres_a
    genre_signal = np.zeros(n, dtype=float)
    if effective_genres:
        genre_bonus = _genre_overlap_arr(df, effective_genres)
        genre_signal = _percentile_rank(genre_bonus)
        total_sim   = 0.82 * total_sim + 0.18 * genre_bonus

    chip_affinity_rank = np.zeros(n, dtype=float)
    if selected_chips:
        chip_affinity_raw = _chip_affinity_arr(df, selected_chips)
        chip_affinity_rank = _percentile_rank(chip_affinity_raw)

    # Genre chip re-ranking: if user explicitly selected genre chips with an anchor,
    # boost films that are predominantly that genre (not just genre-overlap)
    chip_genre_bias = np.ones(n, dtype=float)
    if anchor_idx is not None and query_genres:
        chip_weights = np.array([
            1.0 + 0.4 * (
                len(query_genres & _genre_tokens(g)) /
                max(len(_genre_tokens(g)), 1)
            )
            for g in df["genres"]
        ], dtype=float)
        chip_max = float(chip_weights.max()) if chip_weights.size else 1.0
        chip_genre_bias = chip_weights / max(chip_max, 1e-9)

    sim_norm  = _percentile_rank(total_sim)
    rating_norm = (
        pd.to_numeric(df.get("weighted_rating", df["vote_average"]), errors="coerce")
        .fillna(0).clip(0, 10).values / 10.0
    )

    if genre_only_mode:
        # Chip-only mode (Input 3 only): Heavily prioritize actual genre/mood presence
        if query_genres == {"Romance"}:
            final = 0.30 * sim_norm + 0.35 * chip_affinity_rank + 0.25 * genre_signal + 0.06 * rating_norm + 0.02 * vc_all + 0.02 * fame_all
            print("[DEBUG][HYBRID] genre_only mix (Romance): sim_norm*0.30 + chip_affinity*0.35 + genre_signal*0.25 + rating*0.06 + vc*0.02 + fame*0.02")
        else:
            final = 0.30 * sim_norm + 0.30 * chip_affinity_rank + 0.30 * genre_signal + 0.06 * vc_all + 0.02 * fame_all + 0.02 * rating_norm
            print("[DEBUG][HYBRID] genre_only mix: sim_norm*0.30 + chip_affinity*0.30 + genre_signal*0.30 + vc*0.06 + fame*0.02 + rating*0.02")
    else:
        final = (0.56 * sim_norm + 0.14 * chip_affinity_rank + 0.12 * genre_signal + 0.10 * vc_all + 0.05 * fame_all + 0.03 * rating_norm) * chip_genre_bias
        print("[DEBUG][HYBRID] mix: sim_norm*0.56 + chip_affinity*0.14 + genre_signal*0.12 + vc*0.10 + fame*0.05 + rating*0.03  (chip_bias applied)")

    def _pos(arr): return int(np.count_nonzero(arr > 0))

    # Apply text gate
    if anchor_idx is None and query_text.strip():
        before = _pos(final); final *= text_gate
        _dbg_filter("HYBRID", "text semantic gate", before, _pos(final), "top-20% embed")

        # Free-text keyword overlap boost (for better Input 2 accuracy)
        if free_text_raw and free_text_raw.strip():
            q_terms = _plot_terms(free_text_raw)
            if q_terms:
                kw_overlap = np.array([
                    len(q_terms & _plot_terms(str(row.get("overview", ""))))
                    for _, row in df.iterrows()
                ], dtype=float)
                kw_boost = _percentile_rank(kw_overlap)
                final = 0.80 * final + 0.20 * kw_boost
                print(f"[DEBUG][HYBRID] free-text keyword boost: {len(q_terms)} terms, weight 0.20")

    # Exclude anchor
    if anchor_idx is not None:
        final[anchor_idx] = 0.0
        same = df["title"].fillna("").astype(str).str.strip().str.lower().values == str(df.loc[anchor_idx].get("title","")).strip().lower()
        final[same] = 0.0

    # Language
    lang_mask = np.ones(n, dtype=bool)
    if language_codes:
        before = _pos(final); lang_mask = df["language"].isin(language_codes).values
        final *= lang_mask; _dbg_filter("HYBRID", "language", before, _pos(final), str(language_codes))

    # Decade
    dec_mask = _decade_mask(df, decade_filter)
    before = _pos(final); final *= dec_mask; _dbg_filter("HYBRID", "decade", before, _pos(final))

    # Force 2010+ for free-text/chip-only modes (Input 2 and 3)
    if anchor_idx is None and (query_text.strip() or selected_chips):
        before = _pos(final)
        year_gate = (df["release_year"].fillna(0).values >= 2010)
        final *= year_gate
        _dbg_filter("HYBRID", "year gate (Input 2/3)", before, _pos(final), ">=2010")

    # Genre-only: enforce at least one genre hit
    if genre_only_mode and query_genres:
        tag_ov = np.array([len(query_genres & _genre_tokens(g)) for g in df["genres"]])
        before = _pos(final); final *= (tag_ov >= 1); _dbg_filter("HYBRID", "genre tag", before, _pos(final))

    # Vote quality floor
    va_gate = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0).values >= min_vote_avg
    before = _pos(final); final *= va_gate; _dbg_filter("HYBRID", "vote_avg gate", before, _pos(final))

    nz_gate = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).values > 0
    before = _pos(final); final *= nz_gate; _dbg_filter("HYBRID", "nonzero votes", before, _pos(final))

    if anchor_idx is not None:
        vc_gate = (pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).values > 20) | (franchise_boost >= 0.75)
        before = _pos(final); final *= vc_gate; _dbg_filter("HYBRID", "vote_count floor", before, _pos(final))

    clean_mask = np.array([_is_clean_title(t) for t in df["title"]])
    before = _pos(final); final *= clean_mask; _dbg_filter("HYBRID", "clean title", before, _pos(final))

    if anchor_idx is None and query_text.strip():
        _ensure_reranker()
        if _reranker is not None:
            rerank_pool = min(int(np.count_nonzero(final > 0.0)), max(top_n * 10, 80), 250)
            if rerank_pool > 0:
                rerank_seed = np.argsort(final)[::-1][:rerank_pool]
                rerank_seed = rerank_seed[final[rerank_seed] > 0.0]
                if len(rerank_seed) > 0:
                    avg_fame_in_pool = float(np.mean(fame_all[rerank_seed])) if len(rerank_seed) > 0 else 0.0
                    rerank_query = _build_free_text_reranker_query(query_text, selected_chips, free_text_raw=free_text_raw, avg_fame=avg_fame_in_pool)
                    candidate_texts = [_make_reranker_text(df.iloc[idx]) for idx in rerank_seed]
                    pair_inputs = [(rerank_query, candidate_text) for candidate_text in candidate_texts]
                    rerank_scores = np.asarray(_reranker.predict(pair_inputs, batch_size=16), dtype=float)
                    rerank_rank = _percentile_rank(rerank_scores)
                    final[rerank_seed] = 0.72 * final[rerank_seed] + 0.28 * rerank_rank
                    print(f"[DEBUG][HYBRID] cross-encoder rerank: pool={len(rerank_seed)} avg_fame={avg_fame_in_pool:.3f} weight=0.28")

    # Year window (disable if explicitly looking for classics and anchor is modern)
    if anchor_idx is not None and q_year is not None and not include_old_movies:
        yr_diff = np.abs(df["release_year"].values - q_year)
        windowed = final * (yr_diff <= 15)
        if int(windowed.astype(bool).sum()) >= top_n * 2:
            _dbg_filter("HYBRID", "year window", _pos(final), int(windowed.astype(bool).sum()), "±15yr")
            final = windowed

    # Genre hard filter for anchor queries
    if anchor_idx is not None and q_genres_a:
        g_ov_anch = np.array([len(q_genres_a & _genre_tokens(g)) for g in df["genres"]])
        if query_text.strip() or query_genres:
            gate = (g_ov_anch >= 1) | (franchise_boost >= 0.75)
            before = _pos(final); final *= gate; final += 0.10 * franchise_boost
            _dbg_filter("HYBRID", "anchor genre/franchise", before, _pos(final), "relaxed for mixed query")
        else:
            gate = (g_ov_anch > 2) | (franchise_boost >= 0.75)
            before = _pos(final); final *= gate; final += 0.12 * franchise_boost
            _dbg_filter("HYBRID", "anchor genre/franchise", before, _pos(final))

    if effective_genres and anchor_idx is not None:
        g_ov_eff = np.array([len(effective_genres & _genre_tokens(g)) for g in df["genres"]])
        genre_ok = final * (g_ov_eff >= 1)
        if int(genre_ok.astype(bool).sum()) >= top_n * 2:
            _dbg_filter("HYBRID", "effective genre gate", _pos(final), int(genre_ok.astype(bool).sum()))
            final = genre_ok

    fetch_n  = top_n * 5 if diversify else top_n * 2
    top_idxs = np.argsort(final)[::-1][:fetch_n]
    top_idxs = top_idxs[final[top_idxs] > 0.0]
    print(f"[DEBUG][HYBRID] ranked pool={len(top_idxs)}  (fetch_n={fetch_n})")

    selected = _mmr(top_idxs, final, top_n) if (diversify and _embed_vecs is not None and len(top_idxs) > top_n) else top_idxs[:top_n]
    print(f"[DEBUG][HYBRID] selected={len(selected)}")

    results = []
    for rank, idx in enumerate(selected, 1):
        if idx >= len(df): continue
        row   = df.iloc[idx]
        score = float(final[idx])
        fm    = float(fame_all[idx])

        cand_genres = _genre_tokens(row.get("genres", []))
        cand_kw     = _meaningful_kw(row.get("keywords",[]))
        cand_cast   = set((row.get("cast",[]) or [])[:5])

        sv = float(s_tfidf[idx]) if idx < len(s_tfidf) else 0.0
        ev = float(s_embed[idx]) if idx < len(s_embed) else 0.0
        cv = float(s_cf[idx])    if idx < len(s_cf)    else 0.0
        snv= float(sim_norm[idx])if idx < len(sim_norm) else 0.0
        vc_v = float(vc_all[idx])if idx < len(vc_all)  else 0.0
        rv   = float(rating_norm[idx]) if idx < len(rating_norm) else 0.0

        if genre_only_mode:
            components = [("vote_confidence",vc_v,0.55),("fame",fm,0.30),("rating_norm",rv,0.15)]
        else:
            components = [("sim_norm",snv,0.70),("vote_confidence",vc_v,0.17),("fame",fm,0.10),("rating_norm",rv,0.03)]

        top3 = sorted(components, key=lambda x: x[1]*x[2], reverse=True)[:3]

        print(f"\n[DEBUG][HYBRID][RANK {rank:02d}] {row.get('title','')} (idx={idx})")
        print(f"  score={score:.4f} | vote_avg={float(row.get('vote_average',0)):.2f} | vote_count={int(row.get('vote_count',0))}")
        print(f"  signals: tfidf={sv:.4f}  embed={ev:.4f}  cf={cv:.4f}  sim_norm={snv:.4f}")
        _dbg_weights("HYBRID", components)
        print("  why: " + ", ".join(f"{n}={r*w:.4f}" for n,r,w in top3))
        print(f"  genres       : {_fmt_list(row.get('genres',[]),10)}")
        print(f"  shared_genres: {_fmt_list(sorted(effective_genres & cand_genres),10)}")
        print(f"  shared_kw    : {_fmt_list(sorted(q_kw_a & cand_kw),12)}")
        print(f"  shared_cast  : {_fmt_list(sorted(q_cast_a & cand_cast),8)}")

        results.append(RecommendedMovie(
            movie_id=str(row.get("id",idx)), title=str(row.get("title","")),
            original_title=str(row.get("original_title","")), language=str(row.get("language","")),
            year=int(row.get("release_year",0)), runtime=int(row.get("runtime",0)),
            vote_average=float(row.get("vote_average",0)), vote_count=int(row.get("vote_count",0)),
            genres=list(row.get("genres",[]) or []), director=str(row.get("director","")),
            cast=list(row.get("cast",[]) or []), poster_path=str(row.get("poster_path","")),
            tagline=str(row.get("tagline","")), overview=str(row.get("overview","")),
            budget=int(row.get("budget",0)), revenue=int(row.get("revenue",0)),
            weighted_score=min(score, 1.0), fame_score=fm,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MMR diversification
# ─────────────────────────────────────────────────────────────────────────────

def _mmr(candidates: np.ndarray, scores: np.ndarray, top_n: int, lambda_: float = 0.5) -> np.ndarray:
    if _embed_vecs is None:
        return candidates[:top_n]
    selected, remaining = [], list(candidates)
    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            sel_vecs = _embed_vecs[selected]
            best = max(remaining, key=lambda i: lambda_ * scores[i] - (1 - lambda_) * float((_embed_vecs[i] @ sel_vecs.T).max()))
        selected.append(best)
        remaining.remove(best)
    return np.array(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Chip / query text builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_query_text(free_text: str, selected_chips: list) -> str:
    """
    Build SBERT query from free text + chip intent.
    Genre chips are converted to descriptive intent phrases so they can
    steer semantic ranking without relying on raw tag names.
    """
    parts = []

    chip_intent = _build_chip_intent_text(selected_chips or [])
    if chip_intent:
        parts.append(chip_intent)

    if free_text:
        t = free_text.strip()
        parts.append(t)
        # Emphasis boost: repeat the free-text part if it's substantial
        if len(t) > 20:
            parts.append(t)
    return " ".join(parts).strip()


def _build_free_text_reranker_query(query_text: str, selected_chips: list, free_text_raw: Optional[str] = None, avg_fame: Optional[float] = None) -> str:
    """Build a concise query string for the cross-encoder reranker.
    
    Args:
        avg_fame: Average fame score of candidate pool. If present, adds a popularity hint.
    """
    parts = []

    chip_intent = _build_chip_intent_text(selected_chips or [])
    if chip_intent:
        parts.append(chip_intent)

    source_text = free_text_raw if free_text_raw and free_text_raw.strip() else query_text
    source_text = _fmt_text(source_text, 260)
    if source_text and source_text != "-":
        parts.append(source_text)

    if avg_fame is not None and avg_fame > 0.3:
        popularity_hint = "Well-known or acclaimed films" if avg_fame > 0.6 else "Reasonably popular films"
        parts.append(f"({popularity_hint})")

    return " ".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    collection,                         # kept for API compat (ChromaDB, unused)
    movie_title:    Optional[str],
    free_text:      Optional[str],
    selected_chips: list,
    language_codes: list,
    top_n:          int   = 10,
    min_rating:     float = 5.0,
    decade_filter:  list  = None,
    include_old_movies: bool = False,
    diversify:      bool  = False,
    df:             Optional[pd.DataFrame] = None,
) -> tuple[list[RecommendedMovie], str]:

    global _df

    if not _engine_ready:
        if df is not None:
            build_engine(df)
        else:
            raise RuntimeError("Engine not built. Call build_engine(df) first.")

    if decade_filter is None:
        decade_filter = ["2020s", "2010s", "2000s", "1990s", "Classic (<1990)"]

    # Resolve anchor
    anchor_idx   = None
    anchor_label = None
    lang_hint    = language_codes[0] if len(language_codes) == 1 else None
    title_request_only = bool(movie_title and not (free_text or "").strip() and not (selected_chips or []))

    if movie_title:
        anchor_idx = _find_movie_idx(movie_title, language=lang_hint, exact_only=title_request_only)
        if anchor_idx is not None:
            anchor_label = _df.loc[anchor_idx, "title"]

    # Build query text (genres excluded from embedding query)
    query_text = _build_query_text(free_text or "", selected_chips or [])

    # Genre chips → structural filter set only
    selected_chip_terms = _genre_tokens(selected_chips or [])
    query_genres = selected_chip_terms & SUPPORTED_GENRES

    min_vote_avg = max(5.0, float(min_rating))

    genre_only_mode = bool(
        anchor_idx is None and
        not (free_text or "").strip() and
        bool(selected_chips)
    )

    effective_decade_filter = list(decade_filter or [])
    if anchor_idx is not None:
        anchor_year = int(_df.loc[anchor_idx].get("release_year", 2000))
        new_filter  = _ensure_anchor_decade(effective_decade_filter, anchor_year, include_old_movies=include_old_movies)
        if new_filter != effective_decade_filter:
            print(f"[DEBUG][REQUEST] decade auto-adjusted to include {_decade_label(anchor_year)}")
        effective_decade_filter = new_filter

    title_only_mode = bool(
        anchor_idx is not None and
        not (free_text or "").strip() and
        not (selected_chips or [])
    )
    title_miss_mode = bool(title_request_only and anchor_idx is None)
    fallback_note = ""

    _dbg_header("REQUEST", "Incoming request")
    print(f"[DEBUG][REQUEST] movie_title    : {movie_title or '-'}")
    print(f"[DEBUG][REQUEST] free_text      : {_fmt_text(free_text or '')}")
    print(f"[DEBUG][REQUEST] chips          : {selected_chips or '-'}")
    print(f"[DEBUG][REQUEST] query_text     : {_fmt_text(query_text)}")
    print(f"[DEBUG][REQUEST] query_genres   : {sorted(query_genres) or '-'}")
    print(f"[DEBUG][REQUEST] languages      : {language_codes}")
    print(f"[DEBUG][REQUEST] decades        : {effective_decade_filter}")
    print(f"[DEBUG][REQUEST] top_n/min_rating: {top_n} / {min_vote_avg:.1f}")
    route_label = "semantic(title-only)" if title_only_mode else ("title-miss→sci-fi" if title_miss_mode else "hybrid")
    print(f"[DEBUG][REQUEST] route          : {route_label}")

    if title_miss_mode:
        print("[DEBUG][REQUEST] exact title miss → trying Sci-Fi fame fallback")

    if anchor_idx is not None:
        print(f"[DEBUG][REQUEST] anchor         : idx={anchor_idx}  title={anchor_label}")
        _dbg_movie("Anchor", _df.loc[anchor_idx], full=True)

    if title_only_mode:
        results = _semantic_recommend(
            anchor_idx=anchor_idx, language_codes=language_codes,
            decade_filter=effective_decade_filter, top_n=top_n,
            min_vote_avg=min_vote_avg, year_window=12,
            min_genre_overlap=1, min_vote_count=20,
        )
        if not results:
            print("[DEBUG][REQUEST] semantic empty → hybrid fallback")
            results = _hybrid_recommend(
                anchor_idx=anchor_idx, query_text=query_text,
                query_genres=query_genres, selected_chips=selected_chips or [], language_codes=language_codes,
                decade_filter=effective_decade_filter, top_n=top_n,
                include_old_movies=include_old_movies,
                min_vote_avg=min_vote_avg, genre_only_mode=False, diversify=diversify, free_text_raw=free_text,
            )
    elif title_miss_mode:
        results = _scifi_fame_fallback(
            top_n=top_n,
            min_vote_avg=min_vote_avg,
            language_codes=language_codes,
            decade_filter=effective_decade_filter,
        )
        if results:
            fallback_note = f'Sci-Fi fallback for "{movie_title}"'
            print(f"[DEBUG][REQUEST] Sci-Fi fallback produced {len(results)} recommendations")
        else:
            print("[DEBUG][REQUEST] Sci-Fi fallback empty → hybrid fallback")
            results = _hybrid_recommend(
                anchor_idx=anchor_idx, query_text=query_text,
                query_genres=query_genres, selected_chips=selected_chips or [], language_codes=language_codes,
                decade_filter=effective_decade_filter, top_n=top_n,
                include_old_movies=include_old_movies,
                min_vote_avg=min_vote_avg, genre_only_mode=genre_only_mode, diversify=diversify, free_text_raw=free_text,
            )
    else:
        results = _hybrid_recommend(
            anchor_idx=anchor_idx, query_text=query_text,
            query_genres=query_genres, selected_chips=selected_chips or [], language_codes=language_codes,
            decade_filter=effective_decade_filter, top_n=top_n,
            include_old_movies=include_old_movies,
            min_vote_avg=min_vote_avg, genre_only_mode=genre_only_mode, diversify=diversify, free_text_raw=free_text,
        )

    chip_only_mode = bool(anchor_idx is None and not (free_text or "").strip() and (selected_chips or []))
    if chip_only_mode and query_genres:
        top_score = float(results[0].weighted_score) if results else 0.0
        top_genre_hit = bool(results and (_genre_tokens(results[0].genres) & query_genres))
        if (not results) or (not top_genre_hit) or (top_score < 0.58):
            fallback_results = _chip_exact_fallback(
                selected_genres=query_genres,
                selected_chips=selected_chips or [],
                language_codes=language_codes,
                decade_filter=effective_decade_filter,
                top_n=top_n,
                min_vote_avg=min_vote_avg,
            )
            if fallback_results:
                results = fallback_results
                fallback_note = f'Exact-tag fallback for {" / ".join(sorted(query_genres))}'
                print(f"[DEBUG][REQUEST] chip fallback produced {len(results)} recommendations")

    print(f"[DEBUG][REQUEST] → {len(results)} recommendations")

    parts = []
    if anchor_label:             parts.append(f"Similar to \u201c{anchor_label}\u201d")
    if free_text:
        parts.append(f'Free text vibe: "{_fmt_text(free_text, 260)}"')
    if selected_chips:
        parts.append(f"Chips: {' · '.join(selected_chips)}")
    if fallback_note:
        parts.append(fallback_note)
    if title_miss_mode and not results:
        parts.append(f'Title not found: "{movie_title}"')
    query_summary = "  |  ".join(parts) if parts else "Custom query"

    return results, query_summary


def sort_results(results: list[RecommendedMovie], sort_by: str) -> list[RecommendedMovie]:
    if sort_by == "rating":     return sorted(results, key=lambda m: m.vote_average,  reverse=True)
    if sort_by == "popularity": return sorted(results, key=lambda m: m.vote_count,    reverse=True)
    if sort_by == "newest":     return sorted(results, key=lambda m: m.year,          reverse=True)
    return                             sorted(results, key=lambda m: m.weighted_score, reverse=True)