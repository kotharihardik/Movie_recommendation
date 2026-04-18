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
    render_favourites_sidebar,
    render_settings_sidebar,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("MOVIES_CSV",    "data/movies.csv")
DB_PATH   = os.environ.get("CHROMA_DB_PATH","./chroma_db")


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
        "justification_cache":  {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Load favourites from disk on first access
    if st.session_state.favourites is None:
        st.session_state.favourites = load_favourites()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Page config (must be first Streamlit call) ────────────────
    st.set_page_config(
        page_title="CineMatch India 🎬",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    init_session_state()

    # ── Load data (cached) ────────────────────────────────────────
    with st.spinner("🎬 Starting CineMatch India… (first run builds the index — ~5 min)"):
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
        st.markdown('<p class="sidebar-title">🎬 CineMatch</p>', unsafe_allow_html=True)
        st.caption(f"{collection.count():,} movies indexed")
        st.markdown("---")

        filters  = render_sidebar_filters()
        settings = render_settings_sidebar()
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

    # ── Handle submission ─────────────────────────────────────────
    if submitted:
        if not any([selected_movie, free_text, selected_chips]):
            st.warning(
                "⚠️ Please enter a movie name, describe what you want, "
                "or select at least one genre/mood chip."
            )
        else:
            # ── Retrieval ─────────────────────────────────────────
            with st.spinner("🎬 Scanning 25,000+ films…"):
                results, query_bundle = get_recommendations(
                    collection     = collection,
                    movie_title    = selected_movie,
                    free_text      = free_text,
                    selected_chips = selected_chips,
                    language_codes = filters["language_codes"],
                    top_n          = top_n,
                    min_rating     = filters["min_rating"],
                    decade_filter  = filters["decade_filter"],
                    diversify      = filters["diversify"],
                    df             = df,
                )

            if not results:
                st.warning(
                    "No movies found for your current filters. "
                    "Try lowering the minimum rating or broadening the language/decade selection."
                )
                st.stop()

            # ── Justifications ────────────────────────────────────
            if settings["show_justifications"]:
                api_key = (
                    settings.get("api_key")
                    or os.environ.get("ANTHROPIC_API_KEY")
                )
                with st.spinner("🤖 Writing personalised recommendations…"):
                    results, st.session_state.justification_cache = batch_justify(
                        movies              = results,
                        query_bundle        = query_bundle,
                        api_key             = api_key,
                        justification_cache = st.session_state.justification_cache,
                    )

            # ── Cache results ─────────────────────────────────────
            st.session_state.last_results      = results
            st.session_state.last_query_bundle = query_bundle

    # ── Results or Hero ───────────────────────────────────────────
    if st.session_state.last_results:
        results = st.session_state.last_results

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
