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
import json
import os
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
    "Do not use emojis, icons, decorative symbols, or formulaic template phrases. "
    "Avoid lead-ins like 'If you liked', 'If you enjoyed', or 'Fans of'."
)

_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_BATCH_SIZE = 5
_GEMINI_TIMEOUT_SECONDS = 30

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

    line_1 = f"{intent_px}{adj} {lang} {top_genre.lower()} energy with a {rating:.1f}★ score from {votes:,} fans."
    if director:
        line_2 = f"Directed by {director}, it leans into {adj} storytelling and a strong genre focus."
    else:
        line_2 = f"The {top_genre.lower()} focus and {adj} tone make it a close mood match."
    if top_cast:
        line_3 = f"{top_cast} leads the cast, giving it familiar star power in this genre space."
    else:
        line_3 = f"A solid {lang} pick that fits the tone and pacing you're after."

    lines = [line_1, line_2, line_3]
    return "\n".join(lines)


# ── LLM justification ─────────────────────────────────────────────────────────

def _clean_llm_text(text: str) -> str:
    """Normalize model output into 2-3 short lines."""
    cleaned = strip_emoji(text.strip().strip('"').strip("'"))
    lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 2:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if len(sentences) >= 2:
            lines = sentences[:3]
        elif sentences:
            lines = sentences
        elif not lines:
            lines = [re.sub(r"\s+", " ", cleaned).strip()]

    if len(lines) > 3:
        lines = lines[:3]

    trimmed_lines = []
    for line in lines:
        words = line.split()
        if len(words) > 18:
            line = " ".join(words[:18]).rstrip() + "..."
        trimmed_lines.append(line)

    joined = "\n".join(trimmed_lines)
    if len(joined) > 600:
        joined = joined[:597] + "..."
    return joined


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

Write 2-3 short lines (each line max 18 words) explaining why this movie matches the user's intent.
Rules:
  - Use only the movie facts above and the user's intent.
  - Mention one or two concrete details from the movie facts.
  - Do NOT invent plot details that are not present in the facts.
  - Keep it specific, natural, and conversational.
    - Do NOT start any line with "This movie", "This film", "It", "If you liked", "If you enjoyed", or "Fans of".
    - Put each line on its own line, no bullets or numbering.
    - Output the justification only.'''


def _build_batch_prompt(movies: list, query_bundle: str, compact_prompt: bool = True) -> str:
    max_query_chars = 280 if compact_prompt else 360
    movie_sections = []
    for idx, movie in enumerate(movies, start=1):
        movie_sections.append(f"Movie {idx}:\n{_movie_prompt_block(movie, compact_prompt=compact_prompt)}")
    movie_block = "\n\n".join(movie_sections)
    n_movies = len(movies)
    return f'''User intent:
"{_clip_text(query_bundle, max_query_chars)}"

Movies:
{movie_block}

Write exactly ONE sentence per movie, max 28 words each.
Rules:
  - Use only the movie facts above and the user's intent.
  - Mention one or two concrete details from the movie facts.
  - Do NOT invent plot details that are not present in the facts.
  - Keep it specific, natural, and conversational.
  - Do NOT start any sentence with "This movie", "This film", or "It".
Return exactly {n_movies} lines, in the same order, formatted like:
1. <sentence>
2. <sentence>
Do not include any extra text.'''


def _parse_batch_output(text: str, expected_count: int) -> list[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            if len(data) >= expected_count:
                return data[:expected_count]
    except Exception:
        pass

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    numbered = []
    for line in lines:
        match = re.match(r"^\d+[\).:\-]\s*(.+)$", line)
        if match:
            numbered.append(match.group(1).strip())
    if len(numbered) >= expected_count:
        return numbered[:expected_count]

    if len(lines) >= expected_count:
        return lines[:expected_count]

    raise ValueError("Gemini batch output did not include enough items")


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
    max_output_tokens: int = 400,
    response_mime_type: Optional[str] = None,
) -> str:
    """Call Gemini via the public REST API and return the generated text."""
    # ── DEBUG: Print prompt being sent ────────────────────────────────────────
    print("\n" + "="*80)
    print("📤 GEMINI API REQUEST")
    print("="*80)
    print(f"Model: {model}")
    print(f"\n🔤 SYSTEM PROMPT:\n{_SYSTEM_PROMPT}")
    print(f"\n❓ USER PROMPT:\n{user_prompt}")
    print("="*80)
    
    combined_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
    generation_config = {
        "temperature": 0.7,
        "topP": 0.9,
        "maxOutputTokens": max_output_tokens,
    }
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [
            {
                "parts": [{"text": combined_prompt}],
            }
        ],
        "generationConfig": generation_config,
    }
    url = _GEMINI_ENDPOINT.format(model=model)
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=_GEMINI_TIMEOUT_SECONDS)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("\n" + "!"*80)
        print("🔴 GEMINI API HTTP ERROR")
        print("!"*80)
        print(f"URL: {response.url}")
        print(f"Status code: {response.status_code}")
        try:
            print("Response body:\n", response.text)
        except Exception:
            pass
        print("!"*80 + "\n")
        raise
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
    
    # ── DEBUG: Print response from Gemini ────────────────────────────────────────
    print("\n" + "="*80)
    print("📥 GEMINI API RESPONSE")
    print("="*80)
    print(f"✅ Generated Justification:\n{text}")
    print("="*80 + "\n")
    
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
        raise ValueError("GEMINI_API_KEY is required to generate justifications.")

    cache_namespace = _cache_namespace(api_key, model)

    try:
        print(f"\n🔄 Calling Gemini LLM for: {movie.title} ({movie.movie_id})")
        user_prompt = _build_justification_prompt(movie, query_bundle, compact_prompt=compact_prompt)

        text = _call_gemini(user_prompt, api_key=api_key, model=model)
        cleaned = _clean_llm_text(text)
        print(f"✅ SUCCESS: Gemini returned for {movie.title}")
        print(f"   Final output: {cleaned}\n")
        return cleaned, cache_namespace

    except requests.exceptions.ReadTimeout:
        print(f"❌ GEMINI API TIMEOUT for {movie.title} ({_GEMINI_TIMEOUT_SECONDS}s)")
        return rule_based_justification(movie, query_bundle), cache_namespace
    except Exception as e:
        print(f"❌ GEMINI API FAILED for {movie.title}: {str(e)}")
        raise


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
    
    # ── DEBUG: Print batch info ──────────────────────────────────────────────────
    print(f"\n🎬 Processing {len(movies)} movies for justifications")
    print(f"Query hash: {q_hash}")
    print(f"Backend: {backend_tag}")
    print()

    missing_movies = []
    for movie in movies:
        cache_key = f"{backend_tag}:{movie.movie_id}_{q_hash}"
        if cache_key in justification_cache:
            movie.justification = justification_cache[cache_key]
            movie.justification_source = backend_tag
            print(f"✅ [CACHED] {movie.title} ({movie.movie_id}) - Source: {backend_tag}")
        else:
            missing_movies.append(movie)

    if not missing_movies:
        return movies, justification_cache

    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to generate justifications.")

    for movie in missing_movies:
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
        print(f"🆕 [GEMINI] {movie.title} ({movie.movie_id})")

    return movies, justification_cache
