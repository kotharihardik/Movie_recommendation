"""
app.py
------
CineMatch India — Main Streamlit application entry point.

Run with:
    streamlit run app.py

First startup: builds ChromaDB index from the CSV (~3-6 min on CPU).
Subsequent startups: loads existing index instantly.
"""

import os

import streamlit as st
from dotenv import load_dotenv

# ── Load .env before anything else ───────────────────────────────────────────
load_dotenv()

from data_pipeline     import run_full_pipeline
from recommend_engine  import get_recommendations, sort_results, build_engine
from llm_client        import batch_justify
from favourites        import load_favourites, add_favourite, remove_favourite, save_favourites
from ui_components     import (
    inject_custom_css,
    render_header,
    render_footer,
    render_hero_section,
    render_query_panel,
    render_results_header,
    render_movie_card,
    render_sidebar_filters,
    render_settings_sidebar,
    render_favourites_sidebar,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("MOVIES_CSV",    "data/movies.csv")
DB_PATH   = os.environ.get("CHROMA_DB_PATH","./chroma_db")
DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or "AIzaSyC9ECyyZszSq7r9JRvOdMcakx74yimd6nk"


# ── Cached startup ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def startup():
    """
    One-time startup: load CSV, build (or load) ChromaDB index.
    Cached with st.cache_resource so it runs only once per server process.
    Returns (chromadb.Collection, cleaned pd.DataFrame).
    """
    collection, df = run_full_pipeline(csv_path=DATA_PATH, db_path=DB_PATH)
    build_engine(df)
    return collection, df


# ── Session state init ────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all session state keys exactly once per browser session."""
    defaults = {
        "favourites":           None,        # loaded from disk on first access
        "selected_chips":       set(),
        "last_results":         None,        # List[RecommendedMovie] | None
        "last_query_bundle":    "",
        "last_query_state":     None,
        "last_filter_state":    None,
        "query_event":          "idle",
        "justification_cache":  {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Load favourites from disk on first access
    if st.session_state.favourites is None:
        st.session_state.favourites = load_favourites()


def run_search(collection, df, filters, selected_movie, free_text, selected_chips, top_n, api_key, show_justifications, include_old_movies=False):
    """Execute retrieval + justification for the current query and filters."""
    with st.spinner("Scanning 25,000+ films…"):
        results, query_bundle = get_recommendations(
            collection     = collection,
            movie_title    = selected_movie,
            free_text      = free_text,
            selected_chips = selected_chips,
            language_codes = filters["language_codes"],
            top_n          = top_n,
            min_rating     = 5.0,
            decade_filter  = filters["decade_filter"],
            include_old_movies = include_old_movies,
            diversify      = filters["diversify"],
            df             = df,
        )

    if not results:
        return None, query_bundle

    if show_justifications:
        with st.spinner("Generating Gemini justifications…"):
            results, st.session_state.justification_cache = batch_justify(
                movies              = results,
                query_bundle        = query_bundle,
                api_key             = api_key,
                justification_cache = st.session_state.justification_cache,
            )

    return results, query_bundle


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Page config (must be first Streamlit call) ────────────────
    st.set_page_config(
        page_title="CineMatch India",
        page_icon="CM",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    init_session_state()

    # ── Load data (cached) ────────────────────────────────────────
    with st.spinner("Starting CineMatch India… (first run builds the index — ~5 min)"):
        try:
            collection, df = startup()
        except FileNotFoundError as e:
            st.error(
                f"**Movie data not found.**\n\n{e}\n\n"
                f"Place your TMDB CSV at `{DATA_PATH}` and restart."
            )
            st.stop()
        except Exception as e:
            st.error(f"**Startup error:** {e}")
            st.stop()

    movie_titles = sorted(df["title"].dropna().unique().tolist())

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-title">CineMatch</p>', unsafe_allow_html=True)
        st.caption(f"{collection.count():,} movies indexed")
        st.markdown("---")

        filters  = render_sidebar_filters()
        # settings = render_settings_sidebar(default_api_key=DEFAULT_GEMINI_API_KEY)
        # if not settings["api_key"]:
        #     st.warning("LLM justifications are using the local fallback because no Gemini key is active.")
        
        # Hardcode settings defaults since UI is removed
        settings = {
            "api_key": DEFAULT_GEMINI_API_KEY,
            "show_justifications": True,
            "show_score_bar": True
        }

        removed_id = render_favourites_sidebar(st.session_state.favourites)

        # Handle removal / clear-all
        if removed_id == "__CLEAR_ALL__":
            st.session_state.favourites = []
            save_favourites([])
            st.rerun()
        elif removed_id:
            st.session_state.favourites = remove_favourite(
                st.session_state.favourites, removed_id
            )
            st.rerun()

    # ── Main content ──────────────────────────────────────────────
    render_header()

    # ── Query panel ───────────────────────────────────────────────
    selected_movie, free_text, selected_chips, top_n, submitted = render_query_panel(
        movie_titles
    )

    current_query_state = (
        selected_movie or "",
        (free_text or "").strip(),
        tuple(sorted(selected_chips or [])),
        int(top_n),
    )
    current_filter_state = (
        tuple(filters["language_codes"]),
        tuple(filters["decade_filter"]),
        bool(filters["diversify"]),
    )
    query_event = st.session_state.get("query_event", "idle")
    has_core_input = bool(selected_movie or (free_text or "").strip() or selected_chips)
    auto_search = bool(query_event in {"movie", "chips"} and has_core_input)
    search_executed = False

    if (
        st.session_state.last_results is not None
        and not submitted
        and st.session_state.last_query_state == current_query_state
        and st.session_state.last_filter_state is not None
        and st.session_state.last_filter_state != current_filter_state
        and any(current_query_state[:3])
    ):
        refreshed_results, refreshed_bundle = run_search(
            collection=collection,
            df=df,
            filters=filters,
            selected_movie=selected_movie,
            free_text=free_text,
            selected_chips=selected_chips,
            top_n=top_n,
            include_old_movies=filters.get("include_old_movies", False),
            api_key=settings["api_key"],
            show_justifications=settings["show_justifications"],
        )
        if refreshed_results:
            st.session_state.last_results = refreshed_results
            st.session_state.last_query_bundle = refreshed_bundle
            st.session_state.last_query_state = current_query_state
            st.session_state.last_filter_state = current_filter_state
        else:
            st.session_state.last_results = None
            st.session_state.last_query_bundle = refreshed_bundle or ""
            st.session_state.last_query_state = current_query_state
            st.session_state.last_filter_state = current_filter_state
            st.session_state.query_event = "idle"
            search_executed = True

    # ── Handle submission ─────────────────────────────────────────
    if (submitted or auto_search) and not search_executed:
        if not has_core_input:
            st.warning(
                "Please enter a movie name, describe what you want, "
                "or select at least one genre/mood chip."
            )
        else:
            results, query_bundle = run_search(
                collection=collection,
                df=df,
                filters=filters,
                selected_movie=selected_movie,
                free_text=free_text,
                selected_chips=selected_chips,
                top_n=top_n,
                include_old_movies=filters.get("include_old_movies", False),
                api_key=settings["api_key"],
                show_justifications=settings["show_justifications"],
            )

            if not results:
                st.warning(
                    "No movies found for your current filters. "
                    "Try broadening the language filter or turning on classic movies."
                )
                st.stop()

            # ── Cache results ─────────────────────────────────────
            st.session_state.last_results      = results
            st.session_state.last_query_bundle = query_bundle
            st.session_state.last_query_state   = current_query_state
            st.session_state.last_filter_state  = current_filter_state
            st.session_state.query_event       = "idle"
            search_executed = True

    # ── Results or Hero ───────────────────────────────────────────
    if st.session_state.last_results:
        results = st.session_state.last_results

        if settings["show_justifications"] and any(not getattr(movie, "justification", "") for movie in results):
            with st.spinner("Generating Gemini justifications…"):
                results, st.session_state.justification_cache = batch_justify(
                    movies              = results,
                    query_bundle        = st.session_state.last_query_bundle,
                    api_key             = settings["api_key"],
                    justification_cache = st.session_state.justification_cache,
                )
                st.session_state.last_results = results

        # Sort controls
        sort_by = render_results_header(
            query_summary = st.session_state.last_query_bundle,
            n_results     = len(results),
        )
        results = sort_results(results, sort_by)

        # Modify-search button
        col_mod, _ = st.columns([1, 5])
        with col_mod:
            if st.button("← New Search", key="new_search_btn"):
                st.session_state.last_results = None
                st.rerun()

        st.markdown("---")

        # ── 2-column card grid ────────────────────────────────────
        fav_ids = {f["movie_id"] for f in st.session_state.favourites}

        left_col, right_col = st.columns(2, gap="medium")
        cols = [left_col, right_col]

        for i, movie in enumerate(results):
            with cols[i % 2]:
                save_clicked = render_movie_card(
                    movie              = movie,
                    rank               = i + 1,
                    is_favourite       = movie.movie_id in fav_ids,
                    show_score_bar     = settings["show_score_bar"],
                    show_justification = settings["show_justifications"],
                )

                if save_clicked:
                    if movie.movie_id in fav_ids:
                        st.session_state.favourites = remove_favourite(
                            st.session_state.favourites, movie.movie_id
                        )
                    else:
                        st.session_state.favourites = add_favourite(
                            st.session_state.favourites, movie
                        )
                    st.rerun()

    else:
        render_hero_section(df=df)

    render_footer()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
