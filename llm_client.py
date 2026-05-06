"""
llm_client.py
-------------
Generates personalised "Why you'll love this" justification sentences
per recommended movie using the Gemini API.

If no API key is provided (or the call fails), a deterministic
rule-based fallback produces a reasonable sentence without any
external calls.
"""

import hashlib
import re
from typing import Optional

import requests

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
    "from a knowledgeable friend — enthusiastic, specific, never generic. "
    "Do not use emojis, icons, decorative symbols, or formulaic template phrases."
)

_GEMINI_MODEL = "gemini-1.5-flash"
_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """Remove common emoji and emoji-style symbols from display text."""
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = cleaned.replace("\ufe0f", "").replace("\u200d", "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _clip_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) > limit:
        return cleaned[: max(limit - 3, 0)].rstrip() + "..."
    return cleaned

# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_justification(movie, query_bundle: str = "") -> str:
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
    intent    = _clip_text(query_bundle, 90)
    intent_px = f"For your {intent}, " if intent else ""

    templates = [
        f"{intent_px}this {adj} {lang} {top_genre.lower()} is rated {rating:.1f}★ by {votes:,} fans and feels like a strong match.",
        f"{intent_px}{'Directed by ' + director + ', this' if director else 'This'} {lang} {top_genre.lower()} delivers {adj} storytelling with a {rating:.1f}★ score.",
        f"{intent_px}{'With ' + top_cast + ' in the lead, this' if top_cast else 'This'} {adj} {lang} film has won {votes:,} fans over.",
        f"{intent_px}one of {lang} cinema's best {top_genre.lower()} entries at {rating:.1f}★, with the right mood and energy.",
        f"{intent_px}{'By ' + director + ', ' if director else ''}a {adj} {lang} {top_genre.lower()} that fans rate a strong {rating:.1f}★.",
    ]

    # Deterministic choice based on movie_id for consistency
    movie_id = str(getattr(movie, "movie_id", ""))
    idx = int(movie_id) % len(templates) if movie_id.isdigit() else 0
    return templates[idx]


# ── LLM justification ─────────────────────────────────────────────────────────

def _clean_llm_text(text: str) -> str:
    """Normalize model output into a single clean sentence."""
    cleaned = strip_emoji(text.strip().strip('"').strip("'"))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 220:
        cleaned = cleaned[:217] + "..."
    return cleaned


def _movie_prompt_block(movie, compact_prompt: bool = True) -> str:
    overview_limit = 220 if compact_prompt else 320
    tagline_limit = 100 if compact_prompt else 140
    lines = [
        f"Title: {movie.title}",
    ]
    if movie.original_title and movie.original_title != movie.title:
        lines.append(f"Original title: {movie.original_title}")
    lines.extend([
        f"Year: {movie.year}",
        f"Language: {movie.language}",
        f"Genres: {', '.join(movie.genres[:5]) if movie.genres else 'Unknown'}",
        f"Director: {movie.director}" if movie.director and movie.director != 'Unknown' else None,
        f"Cast: {', '.join(movie.cast[:4]) if movie.cast else 'Unknown'}",
        f"Runtime: {movie.runtime} min" if not compact_prompt and movie.runtime else None,
        f"Rating: {movie.vote_average:.1f}/10 from {movie.vote_count:,} votes",
        f"Tagline: {_clip_text(movie.tagline, tagline_limit)}" if movie.tagline and movie.tagline != 'Unknown' else None,
        f"Overview: {_clip_text(movie.overview, overview_limit)}" if movie.overview else None,
        f"Budget: {movie.budget:,}" if not compact_prompt and movie.budget else None,
        f"Revenue: {movie.revenue:,}" if not compact_prompt and movie.revenue else None,
    ])
    return "\n".join(f"- {line}" for line in lines if line)


def _build_justification_prompt(movie, query_bundle: str, compact_prompt: bool = True) -> str:
    max_query_chars = 280 if compact_prompt else 360
    movie_block = _movie_prompt_block(movie, compact_prompt=compact_prompt)
    return f'''User intent:
"{_clip_text(query_bundle, max_query_chars)}"

Movie facts:
{movie_block}

Write exactly ONE sentence, max 28 words, explaining why this movie matches the user's intent.
Rules:
  - Use only the movie facts above and the user's intent.
  - Mention one or two concrete details from the movie facts.
  - Do NOT invent plot details that are not present in the facts.
  - Keep it specific, natural, and conversational.
  - Do NOT start with "This movie", "This film", or "It".
  - Output the justification only.'''


def _cache_namespace(api_key: Optional[str], model: str = _GEMINI_MODEL) -> str:
    """Namespace cache entries by backend and API key fingerprint."""
    if not api_key:
        return "fallback"
    key_hash = hashlib.md5(api_key.encode()).hexdigest()[:10]
    return f"gemini:{key_hash}:{model}"


def _call_gemini(
    user_prompt: str,
    api_key: str,
    model: str = _GEMINI_MODEL,
) -> str:
    """Call Gemini via the public REST API and return the generated text."""
    payload = {
        "systemInstruction": {
            "parts": [{"text": _SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 120,
        },
    }
    url = _GEMINI_ENDPOINT.format(model=model)
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise ValueError(str(data["error"]))

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError("Gemini returned empty content")
    return text


def get_justification(
    movie,
    query_bundle: str,
    api_key:      Optional[str] = None,
    model:        str = _GEMINI_MODEL,
    compact_prompt: bool = True,
) -> tuple[str, str]:
    """
    Call Gemini to generate one personalised justification sentence.
    Falls back to rule_based_justification on any error or missing key.
    """
    if not api_key:
        return rule_based_justification(movie, query_bundle), "fallback"

    cache_namespace = _cache_namespace(api_key, model)

    try:
        user_prompt = _build_justification_prompt(movie, query_bundle, compact_prompt=compact_prompt)

        text = _call_gemini(user_prompt, api_key=api_key, model=model)
        return _clean_llm_text(text), cache_namespace

    except Exception:
        return rule_based_justification(movie, query_bundle), cache_namespace


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _query_hash(query_bundle: str) -> str:
    """Short MD5 hash of the query for cache keying."""
    return hashlib.md5(query_bundle.encode()).hexdigest()[:10]


def batch_justify(
    movies:               list,
    query_bundle:         str,
    api_key:              Optional[str] = None,
    model:                str = _GEMINI_MODEL,
    justification_cache:  Optional[dict] = None,
) -> tuple:
    """
    Fill in movie.justification for each movie.
    Checks cache keyed by (backend, movie_id, query_hash) before calling the API.

    Returns:
        (updated_movies_list, updated_cache_dict)
    """
    if justification_cache is None:
        justification_cache = {}

    q_hash = _query_hash(query_bundle)
    backend_tag = _cache_namespace(api_key, model)

    for movie in movies:
        cache_key = f"{backend_tag}:{movie.movie_id}_{q_hash}"
        if cache_key in justification_cache:
            movie.justification = justification_cache[cache_key]
            movie.justification_source = backend_tag
        else:
            just, cache_backend = get_justification(
                movie,
                query_bundle,
                api_key=api_key,
                model=model,
                compact_prompt=True,
            )
            movie.justification = just
            movie.justification_source = cache_backend
            justification_cache[f"{cache_backend}:{movie.movie_id}_{q_hash}"] = just

    return movies, justification_cache
