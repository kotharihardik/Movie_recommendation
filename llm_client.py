"""
llm_client.py
-------------
Generates short "Why it matches" justifications for recommended movies.

This version uses the Hugging Face router with a DeepSeek chat model and
batches up to 10 movies per request so the UI can render much faster than a
one-request-per-movie flow.

If no HF token is provided, or the model call fails, a deterministic
rule-based fallback produces a usable explanation without any external calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

from openai import OpenAI

# ── Constants ─────────────────────────────────────────────────────────────────

_LANG_LABEL = {
    "hi": "Bollywood",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}

_GENRE_ADJ = {
    "Action": "adrenaline-pumping",
    "Romance": "heartwarming",
    "Thriller": "edge-of-your-seat",
    "Drama": "emotionally rich",
    "Comedy": "laugh-out-loud",
    "Horror": "spine-chilling",
    "Family": "wholesome",
    "Historical": "grand period",
    "Crime": "gritty",
    "Mystery": "intriguing",
    "Adventure": "thrilling",
    "Biography": "inspiring",
    "Fantasy": "enchanting",
    "War": "powerful",
}

_SYSTEM_PROMPT = (
    "You are a concise Indian cinema recommendation assistant. "
    "Write short, specific, natural explanations grounded only in the provided movie facts and user intent. "
    "Avoid generic template phrases, emojis, bullets, and invented plot details."
)

_HF_MODEL = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-V4-Pro:novita")
_HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
_HF_TIMEOUT_SECONDS = float(os.environ.get("HF_TIMEOUT_SECONDS", "25"))
_BATCH_SIZE = 10

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

_CLIENT: Optional[OpenAI] = None
_CLIENT_TOKEN: Optional[str] = None


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


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_justification(movie, query_bundle: str = "") -> str:
    """Generate a 4-line justification using only the movie's metadata."""
    top_genre = movie.genres[0] if getattr(movie, "genres", None) else "cinematic"
    adj = _GENRE_ADJ.get(top_genre, "compelling")
    lang = _LANG_LABEL.get(getattr(movie, "language", ""), "Indian")
    director = movie.director if getattr(movie, "director", None) and movie.director != "Unknown" else None
    rating = float(getattr(movie, "vote_average", 0.0) or 0.0)
    votes = int(getattr(movie, "vote_count", 0) or 0)
    top_cast = movie.cast[0] if getattr(movie, "cast", None) else None
    intent = _clip_text(query_bundle, 90)
    intent_px = f"For your '{intent}', " if intent else ""

    # Line 1: Genre, language, rating
    line_1 = f"{intent_px}{adj} {lang} {top_genre.lower()} with {rating:.1f}★ from {votes:,} fans."
    
    # Line 2: Director or tone
    if director:
        line_2 = f"Directed by {director}, known for {adj} storytelling and strong narrative focus."
    else:
        line_2 = f"The {top_genre.lower()} focus and {adj} tone create an engaging mood match for you."
    
    # Line 3: Cast or themes
    if top_cast:
        line_3 = f"{top_cast} leads the ensemble, bringing familiar charm and screen presence."
    else:
        line_3 = f"Strong performances lift this {lang} film into compelling entertainment territory."
    
    # Line 4: Summary appeal
    line_4 = f"A solid pick that blends {top_genre.lower()} conventions with genuine emotion and flair."

    return "\n".join([line_1, line_2, line_3, line_4])


# ── Prompt builders ──────────────────────────────────────────────────────────

def _movie_prompt_block(movie, compact_prompt: bool = True) -> str:
    overview_limit = 200 if compact_prompt else 300
    tagline_limit = 90 if compact_prompt else 130
    lines = [f"ID: {getattr(movie, 'movie_id', '')}", f"Title: {movie.title}"]
    if getattr(movie, "original_title", None) and movie.original_title != movie.title:
        lines.append(f"Original title: {movie.original_title}")
    lines.extend(
        [
            f"Year: {getattr(movie, 'year', '')}",
            f"Language: {getattr(movie, 'language', '')}",
            f"Genres: {', '.join(movie.genres[:5]) if getattr(movie, 'genres', None) else 'Unknown'}",
            f"Director: {movie.director}" if getattr(movie, "director", None) and movie.director != "Unknown" else None,
            f"Cast: {', '.join(movie.cast[:4]) if getattr(movie, 'cast', None) else 'Unknown'}",
            f"Rating: {float(getattr(movie, 'vote_average', 0.0) or 0.0):.1f}/10 from {_safe_int(getattr(movie, 'vote_count', 0)):,} votes",
            f"Tagline: {_clip_text(getattr(movie, 'tagline', ''), tagline_limit)}" if getattr(movie, "tagline", None) and movie.tagline != "Unknown" else None,
            f"Overview: {_clip_text(getattr(movie, 'overview', ''), overview_limit)}" if getattr(movie, "overview", None) else None,
        ]
    )
    return "\n".join(f"- {line}" for line in lines if line)


def _build_justification_prompt(movie, query_bundle: str, compact_prompt: bool = True) -> str:
    # Ask the model to use its background knowledge about the movie.
    max_query_chars = 240 if compact_prompt else 320
    title = getattr(movie, "title", "")
    year = getattr(movie, "year", "")
    return f'''User intent:
"{_clip_text(query_bundle, max_query_chars)}"

Recommended movie:
Title: {title}
Year: {year}

Instruction:
Using your knowledge about the movie "{title}" (and the year if helpful), write a concise justification of 3 to 4 short lines explaining why this movie matches the user's intent. You may rely on your background knowledge about the film (genre, notable cast, director, tone, awards or reputation). Avoid inventing specific plot events or false facts. Be natural, specific, and grounded.

Output: a single string made of 3 to 4 short lines separated by newlines.  Do not include extra commentary, markdown, or JSON.'''


def _build_batch_prompt(movies: list, query_bundle: str, compact_prompt: bool = True) -> str:
    max_query_chars = 240 if compact_prompt else 320
    movie_sections = []
    for idx, movie in enumerate(movies, start=1):
        movie_sections.append(f"Movie {idx}:\n{_movie_prompt_block(movie, compact_prompt=compact_prompt)}")
    movie_block = "\n\n".join(movie_sections)
    n_movies = len(movies)

    # Ask the model to use its background knowledge about each recommended
    # title. For speed, pass only title and year so the model relies on its
    # internal knowledge to produce justifications.
    movie_sections = []
    for idx, movie in enumerate(movies, start=1):
        title = getattr(movie, "title", "")
        year = getattr(movie, "year", "")
        year_line = f" (Year: {year})" if year else ""
        movie_sections.append(f"Movie {idx}:\n- Title: {title}{year_line}")
    movie_block = "\n\n".join(movie_sections)

    return f'''User intent:
"{_clip_text(query_bundle, max_query_chars)}"

Recommended movies (title + year):
{movie_block}

Return a JSON array with exactly {n_movies} strings, one string per movie in the same order. Each string must contain 3 to 4 short lines (separated by newlines) that together form a concise justification for why the movie matches the user's intent. You may use your background knowledge about each titled film (genre, notable cast/director, tone, reputation). Avoid inventing specific plot events or false facts.

Rules:
  - Each justification: 3-4 short lines, each line 6-18 words.
  - Prefer well-known facts (genre, director, lead actor, awards, tone).
  - Do NOT invent new plot events or specific fabricated details.
  - Do NOT include emojis, markdown, or commentary.

Output format example (exact JSON array only):
["line1\nline2\nline3", "line1\nline2\nline3\nline4"]'''


# ── Router helpers ────────────────────────────────────────────────────────────

def _get_client(api_key: Optional[str]) -> Optional[OpenAI]:
    global _CLIENT, _CLIENT_TOKEN
    token = (api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    if not token:
        _CLIENT = None
        _CLIENT_TOKEN = None
        return None
    if _CLIENT is None or token != _CLIENT_TOKEN:
        _CLIENT = OpenAI(base_url=_HF_BASE_URL, api_key=token, timeout=_HF_TIMEOUT_SECONDS)
        _CLIENT_TOKEN = token
    return _CLIENT


def _cache_namespace(api_key: Optional[str], model: str = _HF_MODEL) -> str:
    """Namespace cache entries by backend and token fingerprint."""
    token = (api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    if not token:
        return "fallback"
    key_hash = hashlib.md5(token.encode()).hexdigest()[:10]
    return f"hf:{key_hash}:{model}"


def _query_hash(query_bundle: str) -> str:
    """Short MD5 hash of the query for cache keying."""
    return hashlib.md5(query_bundle.encode()).hexdigest()[:10]


def _clean_llm_text(text: str) -> str:
    """Normalize model output while preserving 3-4 short lines.

    Previously this function collapsed output to 1-2 lines which made the
    justifications too short. Now allow up to 4 lines and limit each line to
    a sensible word count to keep outputs concise.
    """
    cleaned = strip_emoji(str(text or "").strip().strip('"').strip("'"))
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Split into non-empty lines and normalize whitespace within each line
    lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        lines = [re.sub(r"\s+", " ", cleaned).strip()]

    # Allow up to 4 lines (prefer 3-4 for richer justification)
    # Keep all lines up to 4 to preserve the full justification
    if len(lines) > 4:
        lines = lines[:4]

    trimmed_lines = []
    for line in lines:
        words = line.split()
        # Keep each line concise; allow up to 24 words then truncate
        if len(words) > 24:
            line = " ".join(words[:24]).rstrip() + "..."
        trimmed_lines.append(line)

    # Preserve newlines between lines
    joined = "\n".join(trimmed_lines).strip()
    # Bound the total length to avoid UI overflow
    max_length = 1500
    if len(joined) > max_length:
        joined = joined[:max_length].rstrip() + "..."
    return joined


def _parse_batch_output(text: str, expected_count: int) -> list[str]:
    """Parse model output as a JSON array of strings (each string may contain newlines)."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, str):
                    # Item is a string (may contain \n for newlines)
                    results.append(item.strip())
                elif isinstance(item, dict):
                    # Try to extract justification from dict keys
                    value = item.get("justification") or item.get("text") or item.get("response") or ""
                    results.append(str(value).strip())
            if len(results) >= expected_count:
                return results[:expected_count]
    except json.JSONDecodeError as e:
        # JSON parsing failed; fall back to line-based extraction
        print(f"  ⚠️  JSON parse failed: {e}. Trying line-based fallback.")

    # Fallback: try to split into roughly equal chunks (one per movie)
    # This assumes the model returned N lines of output.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= expected_count:
        # If we have at least expected_count lines, group them
        chunk_size = len(lines) // expected_count
        results = []
        for i in range(expected_count):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < expected_count - 1 else len(lines)
            chunk = "\n".join(lines[start:end])
            results.append(chunk)
        return results

    raise ValueError(f"DeepSeek batch output could not be parsed (got {len(lines)} lines, need {expected_count})")


def _generate_batch_justifications(
    movies: list,
    query_bundle: str,
    api_key: Optional[str] = None,
    model: str = _HF_MODEL,
    compact_prompt: bool = True,
) -> list[str]:
    client = _get_client(api_key)
    if client is None:
        raise ValueError("HF_TOKEN is required for hosted justifications")

    prompt = _build_batch_prompt(movies, query_bundle, compact_prompt=compact_prompt)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
        max_tokens=min(450, 60 * max(len(movies), 1)),
    )
    content = response.choices[0].message.content or ""
    return _parse_batch_output(content, len(movies))


# ── Public API ───────────────────────────────────────────────────────────────

def get_justification(
    movie,
    query_bundle: str,
    api_key: Optional[str] = None,
    model: str = _HF_MODEL,
    compact_prompt: bool = True,
) -> tuple[str, str]:
    """Generate a single justification using the batched DeepSeek path."""
    backend = _cache_namespace(api_key, model)
    try:
        text = _generate_batch_justifications([movie], query_bundle, api_key=api_key, model=model, compact_prompt=compact_prompt)[0]
        return _clean_llm_text(text), backend
    except Exception as exc:
        # Temporarily disable rule-based fallback — return empty justification
        # so hosted failures are visible during testing.
        print(f"⚠️ Hosted get_justification failed: {exc}")
        return "", "hf_failed"


def batch_justify(
    movies: list,
    query_bundle: str,
    api_key: Optional[str] = None,
    model: str = _HF_MODEL,
    justification_cache: Optional[dict] = None,
) -> tuple:
    """
    Fill in movie.justification for each movie.

    The function batches up to 10 movies per request, reuses cache entries,
    and falls back to local justifications if the hosted model call fails.
    """
    if justification_cache is None:
        justification_cache = {}

    q_hash = _query_hash(query_bundle)
    backend_tag = _cache_namespace(api_key, model)

    print(f"\n🎬 Processing {len(movies)} movies for justifications")
    print(f"Query hash: {q_hash}")
    print(f"Backend: {backend_tag}")
    print()

    pending = []
    for movie in movies:
        movie_id = getattr(movie, "movie_id", movie.title)
        cache_key = f"{backend_tag}:{movie_id}_{q_hash}"
        if cache_key in justification_cache:
            movie.justification = justification_cache[cache_key]
            movie.justification_source = backend_tag
            print(f"✅ [CACHED] {movie.title} ({movie_id}) - Source: {backend_tag}")
        else:
            pending.append(movie)

    if not pending:
        return movies, justification_cache

    if api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        chunks = [pending[i:i + _BATCH_SIZE] for i in range(0, len(pending), _BATCH_SIZE)]
        for chunk in chunks:
            try:
                texts = _generate_batch_justifications(chunk, query_bundle, api_key=api_key, model=model, compact_prompt=True)
            except Exception as exc:
                # Hosted model failed; fall back to local rule-based generation.
                # This ensures UI always has output even if hosted call fails.
                print(f"⚠️ Hosted batch failed, using local fallback: {exc}")
                texts = [rule_based_justification(movie, query_bundle) for movie in chunk]
                backend_tag = "fallback"

            if len(texts) < len(chunk):
                # If any movie is missing a justification, fill with fallback
                texts = list(texts) + [rule_based_justification(movie, query_bundle) for movie in chunk[len(texts):]]

            for movie, text in zip(chunk, texts):
                movie_id = getattr(movie, "movie_id", movie.title)
                cleaned = _clean_llm_text(text)
                movie.justification = cleaned
                movie.justification_source = backend_tag
                justification_cache[f"{backend_tag}:{movie_id}_{q_hash}"] = cleaned
                print(f"🆕 [{backend_tag}] {movie.title} ({movie_id})")
    else:
        # No HF token present — use local rule-based fallback.
        print("⚠️ No HF token found — using local rule-based fallback")
        for movie in pending:
            cleaned = rule_based_justification(movie, query_bundle)
            movie_id = getattr(movie, "movie_id", movie.title)
            movie.justification = cleaned
            movie.justification_source = "fallback"
            justification_cache[f"fallback:{movie_id}_{q_hash}"] = cleaned
            print(f"🆕 [fallback] {movie.title} ({movie_id})")

    return movies, justification_cache
