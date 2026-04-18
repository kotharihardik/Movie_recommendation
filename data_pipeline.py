"""
data_pipeline.py
----------------
Loads the TMDB movies CSV, cleans & filters it, builds rich text documents,
and upserts everything into a ChromaDB persistent collection using the
all-MiniLM-L6-v2 sentence-transformer embedding model.

Run once at startup; subsequent calls skip re-indexing if the DB is populated.
"""

import ast
import math
import os

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "hi": "Hindi (Bollywood)",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}

TARGET_LANGUAGES = list(LANGUAGE_MAP.keys())

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "cinmatch_india_v1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_parse_list(val) -> list:
    """Parse a stringified Python list; return [] on failure."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


# ── Core Functions ────────────────────────────────────────────────────────────

def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """Load the raw TMDB movies CSV. Returns a DataFrame with all original columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Movie CSV not found at '{csv_path}'.\n"
            f"Please place your TMDB dataset CSV at that path."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def clean_and_filter(
    df: pd.DataFrame,
    target_languages: list = None,
) -> pd.DataFrame:
    """
    Drop nulls, parse list columns, deduplicate, filter by language.
    Returns cleaned DataFrame.
    """
    if target_languages is None:
        target_languages = TARGET_LANGUAGES

    # ── Language filter ───────────────────────────────────────────
    df = df[df["language"].isin(target_languages)].copy()

    # ── Drop rows with missing / short overview ───────────────────
    df = df.dropna(subset=["overview"])
    df = df[df["overview"].astype(str).str.strip().str.len() >= 20]

    # ── Deduplicate ───────────────────────────────────────────────
    df = df.drop_duplicates(subset=["id"])

    # ── Parse list columns ────────────────────────────────────────
    for col in ["genres", "keywords", "cast"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Clip cast to 5 names
    df["cast"] = df["cast"].apply(lambda x: x[:5] if isinstance(x, list) else [])

    # ── Fill missing values ───────────────────────────────────────
    df["poster_path"]   = df.get("poster_path",   pd.Series()).fillna("").astype(str)
    df["tagline"]       = df.get("tagline",        pd.Series()).fillna("").astype(str)
    df["director"]      = df.get("director",       pd.Series()).fillna("Unknown").astype(str)
    df["runtime"]       = pd.to_numeric(df.get("runtime",       0), errors="coerce").fillna(0).astype(int)
    df["vote_count"]    = pd.to_numeric(df.get("vote_count",    0), errors="coerce").fillna(0).astype(int)
    df["vote_average"]  = pd.to_numeric(df.get("vote_average",  0), errors="coerce").fillna(0.0)
    df["popularity"]    = pd.to_numeric(df.get("popularity",    0), errors="coerce").fillna(0.0)
    df["budget"]        = pd.to_numeric(df.get("budget",        0), errors="coerce").fillna(0).astype(int)
    df["revenue"]       = pd.to_numeric(df.get("revenue",       0), errors="coerce").fillna(0).astype(int)
    df["release_date"]  = df.get("release_date",   pd.Series()).fillna("2000-01-01").astype(str)
    df["title"]         = df.get("title",          pd.Series()).fillna("Unknown").astype(str)
    df["original_title"]= df.get("original_title", pd.Series()).fillna("").astype(str)

    # ── Extract year ──────────────────────────────────────────────
    df["release_year"] = (
        pd.to_datetime(df["release_date"], errors="coerce")
        .dt.year
        .fillna(2000)
        .astype(int)
    )

    df = df.reset_index(drop=True)
    return df


def build_rich_text(row: pd.Series) -> str:
    """
    Assemble the embedding document string for a single movie row.
    Combines title, tagline, overview, genres, keywords, director, cast, language, year.
    """
    title          = str(row.get("title", ""))
    tagline        = str(row.get("tagline", ""))
    overview       = str(row.get("overview", ""))
    genres         = row.get("genres", [])   or []
    keywords       = row.get("keywords", []) or []
    director       = str(row.get("director", ""))
    cast           = row.get("cast", [])     or []
    language       = str(row.get("language", ""))
    year           = str(row.get("release_year", ""))
    language_full  = LANGUAGE_MAP.get(language, language)

    genres_str   = ", ".join(genres[:6])    if genres    else ""
    keywords_str = ", ".join(keywords[:10]) if keywords  else ""
    cast_str     = ", ".join(cast[:3])      if cast      else ""

    parts = [f"{title}."]
    if tagline:
        parts.append(f"{tagline}.")
    parts.append(overview)
    if genres_str:
        parts.append(f"Genres: {genres_str}.")
    if keywords_str:
        parts.append(f"Keywords: {keywords_str}.")
    if director and director != "Unknown":
        parts.append(f"Director: {director}.")
    if cast_str:
        parts.append(f"Cast: {cast_str}.")
    parts.append(f"Language: {language_full}. Released: {year}.")

    return " ".join(parts)


def build_all_rich_texts(df: pd.DataFrame) -> list:
    """Apply build_rich_text to every row. Returns list aligned with df index."""
    return [build_rich_text(row) for _, row in df.iterrows()]


def get_or_create_chromadb_collection(
    db_path: str = "./chroma_db",
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """
    Initialise (or load existing) ChromaDB persistent client and collection.
    Returns the collection object.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_movies_to_chromadb(
    collection: chromadb.Collection,
    df: pd.DataFrame,
    rich_texts: list,
    batch_size: int = 512,
) -> None:
    """
    Batch-upsert all movies into ChromaDB.
    Document = rich_text string.
    Metadata = dict of display fields (primitive types only).
    """
    total = len(df)

    for start in tqdm(range(0, total, batch_size), desc="📥 Indexing movies"):
        end = min(start + batch_size, total)
        batch_df   = df.iloc[start:end]
        batch_docs = rich_texts[start:end]

        ids       = []
        documents = []
        metadatas = []

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            genres_list = row.get("genres", []) or []
            cast_list   = row.get("cast",   []) or []

            ids.append(str(row["id"]))
            documents.append(batch_docs[idx])
            metadatas.append({
                "title":          str(row.get("title", "")),
                "original_title": str(row.get("original_title", "")),
                "language":       str(row.get("language", "")),
                "vote_average":   float(row.get("vote_average", 0.0)),
                "popularity":     float(row.get("popularity",   0.0)),
                "vote_count":     int(row.get("vote_count",     0)),
                "release_year":   int(row.get("release_year",   2000)),
                "genres":         ",".join(genres_list) if genres_list else "",
                "poster_path":    str(row.get("poster_path",    "")),
                "director":       str(row.get("director",       "")),
                "cast_str":       ",".join(cast_list)   if cast_list   else "",
                "tagline":        str(row.get("tagline",        ""))[:200],
                "runtime":        int(row.get("runtime",        0)),
                "overview":       str(row.get("overview",       ""))[:500],
                "budget":         int(row.get("budget",         0)),
                "revenue":        int(row.get("revenue",        0)),
            })

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def index_is_populated(collection: chromadb.Collection) -> bool:
    """Return True if the collection already has documents (skip re-indexing)."""
    return collection.count() > 0


def run_full_pipeline(
    csv_path: str,
    db_path: str = "./chroma_db",
) -> tuple:
    """
    Full startup pipeline: load → clean → embed → index.
    Skips embedding if ChromaDB is already populated.

    Returns:
        (collection, cleaned_dataframe)
    """
    collection = get_or_create_chromadb_collection(db_path=db_path)
    df = load_raw_csv(csv_path)
    df = clean_and_filter(df)

    if not index_is_populated(collection):
        print(f"Building index for {len(df):,} movies — this only happens once...")
        rich_texts = build_all_rich_texts(df)
        upsert_movies_to_chromadb(collection, df, rich_texts)
        print(f"✅ Index built: {collection.count():,} movies.")
    else:
        print(f"✅ Index already populated: {collection.count():,} movies. Skipping re-indexing.")

    return collection, df
