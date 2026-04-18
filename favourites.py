"""
favourites.py
-------------
Manages a user's saved favourite movies.
Persists to a local JSON file so favourites survive across Streamlit sessions.
"""

import csv
import io
import json
import os
from typing import Optional

FAVOURITES_FILE = "favourites.json"


# ── Persistence ───────────────────────────────────────────────────────────────

def load_favourites() -> list:
    """
    Read favourites.json.
    Returns list of movie dicts; empty list if file doesn't exist or is corrupt.
    """
    if not os.path.exists(FAVOURITES_FILE):
        return []
    try:
        with open(FAVOURITES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def save_favourites(favourites: list) -> None:
    """Write the current favourites list to favourites.json."""
    try:
        with open(FAVOURITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favourites, f, ensure_ascii=False, indent=2)
    except IOError:
        pass  # Silently fail — don't crash the UI for a persistence error


# ── Conversion ────────────────────────────────────────────────────────────────

def movie_to_dict(movie) -> dict:
    """Convert a RecommendedMovie dataclass instance to a plain dict for storage."""
    return {
        "movie_id":      movie.movie_id,
        "title":         movie.title,
        "original_title": movie.original_title,
        "language":      movie.language,
        "year":          movie.year,
        "vote_average":  movie.vote_average,
        "vote_count":    movie.vote_count,
        "director":      movie.director,
        "cast":          movie.cast,
        "genres":        movie.genres,
        "poster_path":   movie.poster_path,
        "tagline":       movie.tagline,
        "runtime":       movie.runtime,
        "overview":      movie.overview[:300],  # truncate for storage
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

def add_favourite(favourites: list, movie) -> list:
    """
    Add movie to favourites if not already present (dedup by movie_id).
    Saves to disk immediately.
    Returns updated list.
    """
    existing_ids = {f["movie_id"] for f in favourites}
    if movie.movie_id not in existing_ids:
        favourites.append(movie_to_dict(movie))
        save_favourites(favourites)
    return favourites


def remove_favourite(favourites: list, movie_id: str) -> list:
    """
    Remove movie from favourites by movie_id.
    Saves to disk immediately.
    Returns updated list.
    """
    favourites = [f for f in favourites if f["movie_id"] != movie_id]
    save_favourites(favourites)
    return favourites


def is_favourite(favourites: list, movie_id: str) -> bool:
    """Return True if movie_id is in the current favourites list."""
    return any(f["movie_id"] == movie_id for f in favourites)


# ── Export ────────────────────────────────────────────────────────────────────

def export_favourites_csv(favourites: list) -> str:
    """
    Convert favourites list to a CSV string suitable for st.download_button.
    Columns: title, original_title, language, year, vote_average, director, genres
    """
    if not favourites:
        return "title,original_title,language,year,vote_average,director,genres\n"

    output = io.StringIO()
    fields = ["title", "original_title", "language", "year", "vote_average", "director", "genres"]
    writer = csv.writer(output)
    writer.writerow(fields)

    for fav in favourites:
        genres_str = ", ".join(fav.get("genres", [])) if isinstance(fav.get("genres"), list) else ""
        writer.writerow([
            fav.get("title",          ""),
            fav.get("original_title", ""),
            fav.get("language",       ""),
            fav.get("year",           ""),
            fav.get("vote_average",   ""),
            fav.get("director",       ""),
            genres_str,
        ])

    return output.getvalue()
