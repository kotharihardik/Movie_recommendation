# 🎬 CineMatch India

> Personalised Bollywood & South Indian Movie Recommender  
> T14.4 — Indian Recommender System

---

## What it does

CineMatch India is a Streamlit app that lets you discover your next favourite movie from a dataset of **25,000+ Bollywood and South Indian films**. You can:

- Search **by movie name** ("Find me movies like *Pushpa*")
- **Describe the vibe** in plain English ("A revenge thriller with folk music and a terrifying villain")
- Click **genre/mood chips** for quick filtering
- Filter by **language** (Hindi / Tamil / Telugu / Malayalam / Kannada)
- Filter by **decade** and **minimum rating**
- Enable **diversity mode** (MMR) to avoid repetitive results
- Get AI-generated **"Why you'll love this"** justifications per result
- **Save favourites** and export them as CSV

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥ 1.32 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent local) |
| LLM Justification | Anthropic Claude Haiku (optional) |
| Dataset | TMDB-sourced CSV (~25,000 films) |
| Poster Images | TMDB Image CDN |

---

## Setup

### 1. Clone / download the project

```bash
cd cinmatch_india
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your movie dataset

Place your TMDB CSV file at:
```
data/movies.csv
```

The CSV must have these columns (all present in the provided dataset):
`id, title, original_title, language, overview, genres, keywords, cast, director,
release_date, runtime, popularity, vote_average, vote_count, budget, revenue, poster_path, tagline`

### 4. (Optional) Add your Anthropic API key

For AI-generated justifications, create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Or paste it into the **⚙️ Settings** panel inside the app.  
Without a key, the app still works using a rule-based fallback.

### 5. Run the app

```bash
streamlit run app.py
```

**First run:** ChromaDB will embed all 25,000 movies using `all-MiniLM-L6-v2` on CPU.  
This takes approximately **3–6 minutes** and only happens once.  
Subsequent runs load the pre-built index instantly.

---

## Project Structure

```
cinmatch_india/
├── app.py               ← Streamlit entry point
├── data_pipeline.py     ← CSV loading, cleaning, ChromaDB indexing
├── recommend_engine.py  ← Retrieval, re-ranking, MMR, sorting
├── llm_client.py        ← Anthropic API + rule-based fallback
├── favourites.py        ← Save/load/export favourites
├── ui_components.py     ← All Streamlit rendering + CSS
├── data/
│   └── movies.csv       ← Your TMDB dataset (place here)
├── chroma_db/           ← Auto-created ChromaDB store
├── favourites.json      ← Auto-created favourites store
├── requirements.txt
└── .env                 ← Optional: ANTHROPIC_API_KEY
```

---

## How it works

1. **Indexing** — Each movie's title, tagline, overview, genres, keywords, director, and cast are combined into a rich text string and embedded with `all-MiniLM-L6-v2`. Vectors are stored in ChromaDB.

2. **Query** — User inputs (movie name + free text + chips) are assembled into one query string and embedded the same way.

3. **Retrieval** — ChromaDB's approximate nearest-neighbour search returns the top-30 closest movies with optional language filter.

4. **Re-ranking** — A composite score combines semantic similarity (55%), vote average (25%), popularity (12%), and vote confidence (8%).

5. **Justification** — Claude Haiku generates one personalised sentence per result explaining why it matches the user's query. Falls back to a rule-based template if no API key is set.

---

## Rubric Compliance

| Requirement | Status |
|---|---|
| Streamlit app | ✅ |
| Text + chip query input | ✅ |
| Ranked card list | ✅ |
| "Why?" per result | ✅ LLM + fallback |
| Save to favourites | ✅ persisted to JSON |
| `all-MiniLM-L6-v2` embeddings | ✅ |
| Cosine similarity retrieval | ✅ ChromaDB |
| LLM justification generation | ✅ Anthropic Claude Haiku |
| No model training | ✅ inference only |
| TMDB dataset filtering | ✅ hi/ta/te/ml/kn |
| Poster images | ✅ TMDB CDN |
| Bollywood / South Indian filter | ✅ sidebar radio |
