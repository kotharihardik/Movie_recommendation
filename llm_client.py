"""
llm_client.py
-------------
Generates personalised "Why you'll love this" justification sentences
per recommended movie using the Anthropic Claude API.

If no API key is provided (or the call fails), a deterministic
rule-based fallback produces a reasonable sentence without any
external calls.
"""

import hashlib
import random
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

_LANG_LABEL = {
    "hi": "Bollywood",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}

_GENRE_ADJ = {
    "Action":    "adrenaline-pumping",
    "Romance":   "heartwarming",
    "Thriller":  "edge-of-your-seat",
    "Drama":     "emotionally rich",
    "Comedy":    "laugh-out-loud",
    "Horror":    "spine-chilling",
    "Family":    "wholesome",
    "Historical": "grand period",
    "Crime":     "gritty",
    "Mystery":   "intriguing",
    "Adventure": "thrilling",
    "Biography": "inspiring",
    "Fantasy":   "enchanting",
    "War":       "powerful",
}

_SYSTEM_PROMPT = (
    "You are a witty, knowledgeable Indian cinema expert. "
    "You write short, personalised movie recommendations that feel like they come "
    "from a knowledgeable friend — enthusiastic, specific, never generic."
)

# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_justification(movie) -> str:
    """
    Generate a template justification using only the movie's metadata.
    No API call required.
    """
    top_genre = movie.genres[0] if movie.genres else "cinematic"
    adj       = _GENRE_ADJ.get(top_genre, "compelling")
    lang      = _LANG_LABEL.get(movie.language, "Indian")
    director  = movie.director if movie.director and movie.director != "Unknown" else None
    rating    = movie.vote_average
    votes     = movie.vote_count
    top_cast  = movie.cast[0] if movie.cast else None

    templates = [
        f"A {adj} {lang} {top_genre.lower()} rated {rating:.1f}★ by {votes:,} fans — exactly the vibe you're after.",
        f"{'Directed by ' + director + ', this' if director else 'This'} {lang} {top_genre.lower()} delivers {adj} storytelling with a {rating:.1f}★ score.",
        f"{'With ' + top_cast + ' in the lead, this' if top_cast else 'This'} {adj} {lang} film has won {votes:,} fans over — and it'll win you too.",
        f"One of {lang} cinema's best {top_genre.lower()} entries at {rating:.1f}★, this checks every box you're looking for.",
        f"{'By ' + director + ', ' if director else ''}a {adj} {lang} {top_genre.lower()} that fans rate a strong {rating:.1f}★ — don't miss it.",
    ]

    # Deterministic choice based on movie_id for consistency
    idx = int(movie.movie_id) % len(templates) if movie.movie_id.isdigit() else 0
    return templates[idx]


# ── LLM justification ─────────────────────────────────────────────────────────

def get_justification(
    movie,
    query_bundle: str,
    api_key:      Optional[str] = None,
    model:        str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Call Claude to generate one personalised justification sentence.
    Falls back to rule_based_justification on any error or missing key.
    """
    if not api_key or not _ANTHROPIC_AVAILABLE:
        return rule_based_justification(movie)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        genres_str   = ", ".join(movie.genres[:4]) if movie.genres else "Drama"
        cast_str     = ", ".join(movie.cast[:3])   if movie.cast   else "Unknown"
        overview_snip = (movie.overview[:150] + "...") if len(movie.overview) > 150 else movie.overview
        lang_label   = _LANG_LABEL.get(movie.language, "Indian")

        user_prompt = f"""User preferences: "{query_bundle}"

Recommended movie: "{movie.title}" ({movie.year}, {lang_label} cinema)
Genres: {genres_str}
Director: {movie.director}
Cast: {cast_str}
Rating: {movie.vote_average:.1f}/10 ({movie.vote_count:,} votes)
Tagline: "{movie.tagline}"
Overview snippet: "{overview_snip}"

Write ONE compelling sentence (max 28 words) explaining why this film matches what the user wants.
Rules:
- Be specific: mention a concrete detail (genre, actor, director, mood, setting)
- Do NOT start with "This movie", "This film", or "This"
- Sound like an enthusiastic knowledgeable friend, not a press release
- Write the justification only — no preamble, no labels"""

        message = client.messages.create(
            model      = model,
            max_tokens = 100,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_prompt}],
        )

        text = message.content[0].text.strip().strip('"').strip("'")
        # Safety: truncate if unexpectedly long
        if len(text) > 220:
            text = text[:217] + "..."
        return text

    except Exception:
        return rule_based_justification(movie)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _query_hash(query_bundle: str) -> str:
    """Short MD5 hash of the query for cache keying."""
    return hashlib.md5(query_bundle.encode()).hexdigest()[:10]


def batch_justify(
    movies:               list,
    query_bundle:         str,
    api_key:              Optional[str] = None,
    justification_cache:  Optional[dict] = None,
) -> tuple:
    """
    Fill in movie.justification for each movie.
    Checks cache keyed by (movie_id, query_hash) before calling the API.

    Returns:
        (updated_movies_list, updated_cache_dict)
    """
    if justification_cache is None:
        justification_cache = {}

    q_hash = _query_hash(query_bundle)

    for movie in movies:
        cache_key = f"{movie.movie_id}_{q_hash}"
        if cache_key in justification_cache:
            movie.justification = justification_cache[cache_key]
        else:
            just = get_justification(movie, query_bundle, api_key=api_key)
            movie.justification = just
            justification_cache[cache_key] = just

    return movies, justification_cache
