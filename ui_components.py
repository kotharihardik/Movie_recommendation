"""
ui_components.py
----------------
All Streamlit rendering functions and custom CSS for CineMatch India.
Dark cinema theme: deep black backgrounds, IMDB-gold accents,
teal for Hindi / crimson for South Indian language badges.
"""

import random
from typing import Optional

import streamlit as st

from favourites import export_favourites_csv

# ── Constants ─────────────────────────────────────────────────────────────────

GENRE_CHIPS = [
    "Action", "Romance", "Thriller", "Drama", "Comedy",
    "Horror", "Family", "Historical", "Crime", "Sci-Fi",
]
MOOD_CHIPS = [
    "Revenge", "Feel-Good", "Tear-Jerker", "Dance",
    "Epic", "Road Trip", "Suspense", "Inspirational",
]

LANGUAGE_INFO = {
    "hi": {"emoji": "🟢", "label": "Hindi",     "badge_css": "lang-hi"},
    "ta": {"emoji": "🔴", "label": "Tamil",     "badge_css": "lang-ta"},
    "te": {"emoji": "🟠", "label": "Telugu",    "badge_css": "lang-te"},
    "ml": {"emoji": "🔵", "label": "Malayalam", "badge_css": "lang-ml"},
    "kn": {"emoji": "🟣", "label": "Kannada",   "badge_css": "lang-kn"},
}

TMDB_BASE  = "https://image.tmdb.org/t/p"
PLACEHOLDER = "https://placehold.co/185x278/1a1a2e/F5C518?text=No+Poster"

EXAMPLE_QUERIES = [
    "A slow-burn revenge thriller with incredible folk music and a terrifying villain",
    "A heartwarming family drama about cultural clashes and forbidden love",
    "An epic action blockbuster with jaw-dropping stunts and a hero's journey",
    "A psychological mystery that keeps you guessing until the very last second",
    "A funny, light-hearted college romance with catchy songs and great chemistry",
    "A gritty gangster saga set in urban streets with moral ambiguity",
    "An emotional father-daughter story that will make you cry and laugh",
]


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    """Inject dark cinema theme CSS."""
    st.markdown("""
    <style>
    /* ── Google Fonts ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

    /* ── Global Reset / Dark Background ───────────────────────── */
    html, body, .stApp {
        background-color: #080808 !important;
        color: #DADADA !important;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0F0F0F !important;
        border-right: 1px solid #1e1e1e;
    }
    [data-testid="stSidebar"] * { color: #DADADA !important; }

    /* ── Header ────────────────────────────────────────────────── */
    .cm-header {
        background: linear-gradient(135deg, #0d0d1a 0%, #0f1729 60%, #0d1a33 100%);
        border-bottom: 3px solid #F5C518;
        padding: 22px 28px 18px;
        border-radius: 10px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .cm-header::before {
        content: "🎬";
        position: absolute;
        right: 24px; top: 50%;
        transform: translateY(-50%);
        font-size: 4em;
        opacity: 0.08;
    }
    .cm-header-title {
        font-family: 'Bebas Neue', cursive;
        color: #F5C518;
        font-size: 2.8em;
        letter-spacing: 4px;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 30px rgba(245,197,24,0.35);
    }
    .cm-header-sub {
        color: #888;
        font-size: 0.88em;
        margin: 6px 0 0;
        letter-spacing: 0.5px;
    }
    .cm-header-stats {
        display: inline-flex;
        gap: 16px;
        margin-top: 8px;
    }
    .cm-stat {
        background: rgba(245,197,24,0.1);
        border: 1px solid rgba(245,197,24,0.2);
        color: #F5C518;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
    }

    /* ── Query Panel ────────────────────────────────────────────── */
    .cm-query-panel {
        background: #111118;
        border: 1px solid #222233;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 22px;
    }

    /* ── Section Labels ─────────────────────────────────────────── */
    .cm-label {
        color: #F5C518;
        font-size: 0.72em;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: block;
    }

    /* ── Movie Cards ────────────────────────────────────────────── */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #222233 !important;
        border-radius: 14px !important;
        background: #111118 !important;
        padding: 2px !important;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #F5C518 !important;
        box-shadow: 0 0 24px rgba(245,197,24,0.08) !important;
    }

    /* ── Rank Badges ────────────────────────────────────────────── */
    .rank-1  { color: #FFD700; font-family:'Bebas Neue',cursive; font-size:1.1em; letter-spacing:1px; }
    .rank-2  { color: #C0C0C0; font-family:'Bebas Neue',cursive; font-size:1.0em; }
    .rank-3  { color: #CD7F32; font-family:'Bebas Neue',cursive; font-size:1.0em; }
    .rank-n  { color: #555;    font-family:'Bebas Neue',cursive; font-size:0.95em; }

    /* ── Language & Genre Chips ─────────────────────────────────── */
    .chip {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 20px;
        font-size: 0.72em;
        font-weight: 600;
        margin: 2px 3px 2px 0;
        letter-spacing: 0.3px;
    }
    .chip-genre { background: #1e1e30; color: #9999cc; border: 1px solid #2e2e4a; }
    .lang-hi  { background: #0d2010; color: #6ee06e; border: 1px solid #1e4020; }
    .lang-ta  { background: #201010; color: #e07070; border: 1px solid #401e1e; }
    .lang-te  { background: #201808; color: #e0a060; border: 1px solid #403010; }
    .lang-ml  { background: #081820; color: #60a0e0; border: 1px solid #103040; }
    .lang-kn  { background: #180820; color: #c070e0; border: 1px solid #301040; }
    .chip-year    { background: #181818; color: #888; border: 1px solid #282828; }
    .chip-runtime { background: #181818; color: #888; border: 1px solid #282828; }

    /* ── Stars ──────────────────────────────────────────────────── */
    .stars    { color: #F5C518; font-size: 0.95em; letter-spacing: 1px; }
    .rating-n { color: #F5C518; font-weight: 700; font-size: 1.05em; }
    .vote-c   { color: #666;    font-size: 0.82em; }

    /* ── Justification Box ──────────────────────────────────────── */
    .just-box {
        background: linear-gradient(135deg, #0d0d1a 0%, #0f1322 100%);
        border-left: 3px solid #F5C518;
        border-radius: 0 8px 8px 0;
        padding: 9px 14px;
        margin-top: 10px;
        font-size: 0.88em;
        color: #ccc;
        font-style: italic;
        line-height: 1.55;
    }

    /* ── Match Score Bar ────────────────────────────────────────── */
    .score-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
    }
    .score-track {
        flex: 1;
        background: #1e1e1e;
        border-radius: 4px;
        height: 5px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 4px;
    }
    .score-lbl { font-size: 0.70em; color: #555; white-space: nowrap; }

    /* ── Poster ─────────────────────────────────────────────────── */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6) !important;
    }

    /* ── Hero Section ───────────────────────────────────────────── */
    .hero-wrap {
        text-align: center;
        padding: 30px 10px 10px;
    }
    .hero-title {
        font-family: 'Bebas Neue', cursive;
        color: #F5C518;
        font-size: 3.2em;
        letter-spacing: 5px;
        text-shadow: 0 0 40px rgba(245,197,24,0.25);
        margin-bottom: 8px;
    }
    .hero-sub { color: #666; font-size: 1em; margin-bottom: 28px; }

    /* ── Streamlit widget overrides ─────────────────────────────── */
    .stTextInput  > div > div > input,
    .stTextArea   > div > div > textarea {
        background: #111118 !important;
        color: #DDD !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 8px !important;
    }
    .stTextInput  > div > div > input:focus,
    .stTextArea   > div > div > textarea:focus {
        border-color: #F5C518 !important;
        box-shadow: 0 0 0 2px rgba(245,197,24,0.15) !important;
    }
    .stSelectbox > div > div {
        background: #111118 !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] {
        background: #0d0d14 !important;
        border: 1px solid #1e1e2e !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] summary { color: #888 !important; }

    /* ── Info / Warning / Success boxes ────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* ── Sidebar title ──────────────────────────────────────────── */
    .sidebar-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.4em;
        color: #F5C518;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }

    /* ── Favourite item ─────────────────────────────────────────── */
    .fav-row {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 7px;
        padding: 7px 10px;
        margin-bottom: 5px;
        font-size: 0.83em;
        color: #ccc;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .cm-footer {
        text-align: center;
        color: #2e2e2e;
        font-size: 0.78em;
        padding: 24px 0 10px;
        border-top: 1px solid #111;
        margin-top: 48px;
    }
    .cm-footer strong { color: #444; }

    /* ── Scrollbar ──────────────────────────────────────────────── */
    ::-webkit-scrollbar        { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track  { background: #080808; }
    ::-webkit-scrollbar-thumb  { background: #222; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* ── Tabs ────────────────────────────────────────────────────── */
    .stTabs [data-testid="stTabsHeader"] {
        background: transparent !important;
        border-bottom: 1px solid #1e1e2e !important;
    }
    .stTabs [data-testid="stTabsHeader"] button {
        color: #666 !important;
        font-weight: 600;
    }
    .stTabs [data-testid="stTabsHeader"] button[aria-selected="true"] {
        color: #F5C518 !important;
        border-bottom: 2px solid #F5C518 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Header / Footer ───────────────────────────────────────────────────────────

def render_header() -> None:
    """Render the CineMatch India banner."""
    st.markdown("""
    <div class="cm-header">
        <div class="cm-header-title">🎬 CineMatch India</div>
        <div class="cm-header-sub">Your personal Bollywood & South Indian movie recommender</div>
        <div class="cm-header-stats">
            <span class="cm-stat">25,000+ Movies</span>
            <span class="cm-stat">5 Languages</span>
            <span class="cm-stat">AI Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    return


# ── Hero Section ──────────────────────────────────────────────────────────────

def render_hero_section(df=None) -> None:
    """Welcome / empty-state section shown before any query is submitted."""
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">LIGHTS. CAMERA. RECOMMEND.</div>
        <div class="hero-sub">
            Tell us a movie you loved, describe the vibe you want,
            or just pick a mood — we'll find your next favourite.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show 4 popular movies as teasers
    if df is not None and len(df) > 0:
        st.markdown(
            "<p style='text-align:center;color:#444;font-size:0.85em;margin-bottom:10px'> Popular right now</p>",
            unsafe_allow_html=True,
        )
        try:
            sample = (
                df[df["poster_path"].str.len() > 5]
                .nlargest(300, "popularity")
                .sample(min(4, len(df)))
            )
            cols = st.columns(4)
            for i, (_, row) in enumerate(sample.iterrows()):
                with cols[i]:
                    st.image(
                        get_poster_url(str(row.get("poster_path", "")), "w185"),
                        use_container_width=True,
                    )
                    lang = LANGUAGE_INFO.get(str(row.get("language", "")), {})
                    lang_emoji = lang.get("emoji", "🎬")
                    st.caption(
                        f"{lang_emoji} **{row['title']}**  \n"
                        f"{row.get('release_year', '')} · ★ {row.get('vote_average', 0):.1f}"
                    )
        except Exception:
            pass

    st.markdown("<br/>", unsafe_allow_html=True)
    example = random.choice(EXAMPLE_QUERIES)
    st.info(f'**Try describing:** *"{example}"*')


# ── Query Panel ───────────────────────────────────────────────────────────────

def render_query_panel(movie_titles: list) -> tuple:
    """
    Render the three-mode query panel.
    Returns: (selected_movie, free_text, selected_chips, top_n, submitted)
    """
    # ── Chip state init ───────────────────────────────────────────
    if "selected_chips" not in st.session_state:
        st.session_state.selected_chips = set()

    st.markdown('<div class="cm-label">Tell us what you want to watch</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="margin:0 0 10px;color:#cfcfcf;font-size:0.94em">'
        '<strong>By Movie Name</strong> (Optional) &nbsp;•&nbsp; '
        '<strong>Describe the Vibe</strong> (Optional) &nbsp;•&nbsp; '
        '<strong>Mood & Genre</strong> (Optional)'
        '</p>',
        unsafe_allow_html=True,
    )
    st.caption("Use any one option or combine all three. Add at least one input before searching.")

    # ── 1) Movie Name (Optional) ────────────────────────────────
    st.markdown("**1) By Movie Name (Optional)**")
    st.caption("Pick a movie you loved and we'll find similar ones.")
    options = [""] + sorted(movie_titles)
    selected_movie_val = st.selectbox(
        "Find movies similar to...",
        options=options,
        index=0,
        key="movie_selectbox",
        label_visibility="collapsed",
    )
    if selected_movie_val:
        st.success(f"Searching for movies similar to **{selected_movie_val}**")

    # ── 2) Free Text (Optional) ─────────────────────────────────
    st.markdown("**2) Describe the Vibe (Optional)**")
    example = random.choice(EXAMPLE_QUERIES)
    st.caption("Describe the mood, tone, setting, or story you're craving.")
    free_text_val = st.text_area(
        "Describe what you're looking for",
        placeholder=f'e.g. "{example}"',
        height=95,
        max_chars=350,
        key="free_text_input",
        label_visibility="collapsed",
    )
    if free_text_val:
        char_c = len(free_text_val)
        color  = "#F5C518" if char_c < 280 else "#e05050"
        st.markdown(
            f'<p style="text-align:right;color:{color};font-size:0.72em;margin-top:-4px">{char_c}/350</p>',
            unsafe_allow_html=True,
        )

    # ── 3) Mood & Genre (Optional) ──────────────────────────────
    st.markdown("**3) Mood & Genre (Optional)**")
    st.caption("Click to toggle genres and moods (multi-select).")

    st.markdown("**Genres**")
    g_cols = st.columns(len(GENRE_CHIPS))
    for i, chip in enumerate(GENRE_CHIPS):
        with g_cols[i]:
            sel = chip in st.session_state.selected_chips
            if st.button(
                chip,
                key=f"g_{chip}",
                type="primary" if sel else "secondary",
                use_container_width=True,
            ):
                if sel:
                    st.session_state.selected_chips.discard(chip)
                else:
                    st.session_state.selected_chips.add(chip)
                st.rerun()

    st.markdown("**Moods**")
    m_cols = st.columns(len(MOOD_CHIPS))
    for i, chip in enumerate(MOOD_CHIPS):
        with m_cols[i]:
            sel = chip in st.session_state.selected_chips
            if st.button(
                chip,
                key=f"m_{chip}",
                type="primary" if sel else "secondary",
                use_container_width=True,
            ):
                if sel:
                    st.session_state.selected_chips.discard(chip)
                else:
                    st.session_state.selected_chips.add(chip)
                st.rerun()

    if st.session_state.selected_chips:
        chips_html = " ".join(
            f'<span class="chip chip-genre">{c}</span>'
            for c in sorted(st.session_state.selected_chips)
        )
        st.markdown(
            f"<div style='margin-top:8px'><strong>Selected:</strong> {chips_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Submit row ────────────────────────────────────────────────
    st.markdown("<div style='margin-top:14px'>", unsafe_allow_html=True)
    col_n, col_btn, col_clear = st.columns([1.4, 2.5, 1])

    with col_n:
        top_n = st.select_slider(
            "Results",
            options=[5, 10, 15, 20],
            value=10,
            key="top_n_slider",
        )

    # Read current values (set by widgets above)
    sm   = st.session_state.get("movie_selectbox", "") or ""
    ft   = st.session_state.get("free_text_input",  "") or ""
    chps = st.session_state.selected_chips
    has_input = bool(sm or ft or chps)

    with col_btn:
        st.markdown("<div style='padding-top:22px'>", unsafe_allow_html=True)
        submitted = st.button(
            "Find My Movies",
            type="primary",
            use_container_width=True,
            disabled=not has_input,
            key="submit_btn",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_clear:
        st.markdown("<div style='padding-top:22px'>", unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True, key="clear_query"):
            st.session_state.selected_chips = set()
            st.session_state.pop("movie_selectbox", None)
            st.session_state.pop("free_text_input",  None)
            st.session_state.last_results = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    return (
        sm or None,
        ft or None,
        list(chps),
        top_n,
        submitted,
    )


# ── Results Header ────────────────────────────────────────────────────────────

def render_results_header(query_summary: str, n_results: int) -> str:
    """
    Render results count + sort controls.
    Returns the selected sort_by key string.
    """
    col_info, col_sort = st.columns([3, 1.4])

    with col_info:
        snippet = query_summary[:90] + ("..." if len(query_summary) > 90 else "")
        st.markdown(
            f'<div style="padding:6px 0">'
            f'<span style="color:#F5C518;font-weight:700;font-size:1.05em">{n_results} results</span>'
            f'<span style="color:#555;font-size:0.82em"> &nbsp;·&nbsp; <em>{snippet}</em></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_sort:
        sort_label = st.selectbox(
            "Sort",
            options=["Best Match", "Rating", "Popularity", "Newest"],
            key="sort_by_select",
            label_visibility="collapsed",
        )

    label_to_key = {
        "Best Match": "best_match",
        "Rating":     "rating",
        "Popularity": "popularity",
        "Newest":     "newest",
    }
    return label_to_key.get(sort_label, "best_match")


# ── Card helpers ──────────────────────────────────────────────────────────────

def render_stars(vote_average: float) -> str:
    """Convert 0–10 vote_average to 5-star Unicode string (rounds to nearest 0.5)."""
    half_stars = round(vote_average / 2 * 2) / 2
    full  = int(half_stars)
    half  = 1 if (half_stars - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("⯨" * half) + "☆" * empty


def format_runtime(minutes: int) -> str:
    """Convert integer minutes to 'Xh Ym' display string."""
    if not minutes or minutes <= 0:
        return "N/A"
    h, m = divmod(int(minutes), 60)
    if h == 0:  return f"{m}m"
    if m == 0:  return f"{h}h"
    return f"{h}h {m}m"


def get_poster_url(poster_path: str, size: str = "w300") -> str:
    """Build full TMDB CDN URL from poster_path suffix."""
    if not poster_path or not poster_path.strip():
        return PLACEHOLDER
    path = poster_path if poster_path.startswith("/") else f"/{poster_path}"
    return f"{TMDB_BASE}/{size}{path}"


def get_language_display(code: str) -> str:
    """'hi' → '🟢 Hindi'"""
    info = LANGUAGE_INFO.get(code, {"emoji": "🎬", "label": code.upper()})
    return f"{info['emoji']} {info['label']}"


def _score_bar_html(weighted_score: float) -> str:
    """Return HTML for the coloured match-score progress bar."""
    pct   = int(weighted_score * 100)
    if pct >= 80:    color = "#4CAF50"
    elif pct >= 60:  color = "#FFC107"
    else:            color = "#F44336"
    return (
        f'<div class="score-row">'
        f'<span class="score-lbl">Match</span>'
        f'<div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="score-lbl">{pct}%</span>'
        f'</div>'
    )


# ── Movie Card ────────────────────────────────────────────────────────────────

def render_movie_card(
    movie,
    rank:              int,
    is_favourite:      bool,
    show_score_bar:    bool = True,
    show_justification: bool = True,
) -> bool:
    """
    Render a single full movie result card.
    Returns True if the save/unsave button was clicked.
    """
    save_clicked = False

    with st.container(border=True):

        # ── Rank badge + Save button ──────────────────────────────
        col_rank, col_save = st.columns([9, 1])
        with col_rank:
            rank_css = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-n")
            rank_txt = {1: "🥇 #1  BEST MATCH", 2: "🥈 #2", 3: "🥉 #3"}.get(rank, f"#{rank}")
            st.markdown(f'<span class="{rank_css}">{rank_txt}</span>', unsafe_allow_html=True)
        with col_save:
            heart = "❤️" if is_favourite else "🤍"
            tip   = "Remove from favourites" if is_favourite else "Save to favourites"
            if st.button(heart, key=f"save_{movie.movie_id}_{rank}", help=tip):
                save_clicked = True

        # ── Poster + Info ─────────────────────────────────────────
        col_poster, col_info = st.columns([1, 3])

        with col_poster:
            st.image(
                get_poster_url(movie.poster_path, "w185"),
                use_container_width=True,
            )

        with col_info:
            # Title
            st.markdown(
                f"<h3 style='margin:0 0 2px;font-family:\"Bebas Neue\",cursive;letter-spacing:1px;"
                f"color:#EEE;font-size:1.5em'>{movie.title}</h3>",
                unsafe_allow_html=True,
            )
            if movie.original_title and movie.original_title != movie.title:
                st.caption(movie.original_title)

            # Language + Year + Runtime chips
            lang_info = LANGUAGE_INFO.get(movie.language, {"badge_css": "lang-hi", "label": "Hindi", "emoji": "🎬"})
            lang_css  = lang_info["badge_css"]
            st.markdown(
                f'<span class="chip {lang_css}">{lang_info["emoji"]} {lang_info["label"]}</span>'
                f'<span class="chip chip-year">📅 {movie.year}</span>'
                f'<span class="chip chip-runtime">⏱ {format_runtime(movie.runtime)}</span>',
                unsafe_allow_html=True,
            )

            # Star rating
            stars = render_stars(movie.vote_average)
            st.markdown(
                f'<span class="stars">{stars}</span>&nbsp;'
                f'<span class="rating-n">{movie.vote_average:.1f}</span>&nbsp;'
                f'<span class="vote-c">({movie.vote_count:,} votes)</span>',
                unsafe_allow_html=True,
            )

            # Genre chips
            if movie.genres:
                genre_html = "".join(
                    f'<span class="chip chip-genre">{g}</span>'
                    for g in movie.genres[:5]
                )
                st.markdown(genre_html, unsafe_allow_html=True)

            # Director + Cast
            if movie.director and movie.director != "Unknown":
                st.markdown(
                    f'<p style="margin:6px 0 2px;font-size:0.82em;color:#888">'
                    f'🎥 <strong style="color:#aaa">{movie.director}</strong></p>',
                    unsafe_allow_html=True,
                )
            if movie.cast:
                cast_str = ", ".join(movie.cast[:3])
                st.markdown(
                    f'<p style="margin:0;font-size:0.80em;color:#666">👤 {cast_str}</p>',
                    unsafe_allow_html=True,
                )

        # ── Match Score Bar ───────────────────────────────────────
        if show_score_bar:
            st.markdown(_score_bar_html(movie.weighted_score), unsafe_allow_html=True)

        # ── Why you'll love this ──────────────────────────────────
        if show_justification and movie.justification:
            st.markdown(
                f'<div class="just-box">🤖 &nbsp;{movie.justification}</div>',
                unsafe_allow_html=True,
            )

        # ── More Details expander ─────────────────────────────────
        with st.expander("▼ More Details"):
            if movie.tagline:
                st.markdown(
                    f'<p style="color:#888;font-style:italic;margin-bottom:10px">🎭 &ldquo;{movie.tagline}&rdquo;</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**📖 Overview**\n\n{movie.overview}")

            if movie.budget or movie.revenue:
                c1, c2 = st.columns(2)
                with c1:
                    if movie.budget > 0:
                        label = f"${movie.budget/1e6:.0f}M" if movie.budget > 1e8 else f"₹{movie.budget/1e7:.1f}Cr"
                        st.metric("💰 Budget", label)
                with c2:
                    if movie.revenue > 0:
                        label = f"${movie.revenue/1e6:.0f}M" if movie.revenue > 1e8 else f"₹{movie.revenue/1e7:.1f}Cr"
                        st.metric("💵 Revenue", label)

    return save_clicked


# ── Sidebar Filters ───────────────────────────────────────────────────────────

def render_sidebar_filters() -> dict:
    """Render all sidebar filter widgets. Returns a filter config dict."""
    st.markdown('<div class="cm-label">Language</div>', unsafe_allow_html=True)
    lang_choice = st.radio(
        "Language",
        options=["Both", "Bollywood (Hindi)", "South Indian", "Custom"],
        index=1,
        key="lang_radio",
        label_visibility="collapsed",
    )

    if "Both" in lang_choice:
        language_codes = ["hi", "ta", "te", "ml", "kn"]
    elif "Bollywood" in lang_choice:
        language_codes = ["hi"]
    elif "South Indian" in lang_choice:
        language_codes = ["ta", "te", "ml", "kn"]
    else:
        lang_map = {"Hindi 🟢": "hi", "Tamil 🔴": "ta", "Telugu 🟠": "te", "Malayalam 🔵": "ml", "Kannada 🟣": "kn"}
        selected = st.multiselect(
            "Choose languages",
            options=list(lang_map.keys()),
            default=list(lang_map.keys()),
            key="lang_multi",
            label_visibility="collapsed",
        )
        language_codes = [lang_map[l] for l in selected] or ["hi", "ta", "te", "ml", "kn"]

    st.markdown("---")
    st.markdown('<div class="cm-label">Old Movies</div>', unsafe_allow_html=True)
    include_old_movies = st.toggle(
        "Include old movies (1990s / classic)",
        value=False,
        key="include_old_movies",
        help="Off: use recent years by default. On: include older decades.",
    )

    st.markdown("---")
    st.markdown('<div class="cm-label">📅 Decade</div>', unsafe_allow_html=True)
    all_decades = ["2020s", "2010s", "2000s", "1990s", "Classic (<1990)"]
    default_decades = all_decades if include_old_movies else ["2020s", "2010s"]
    decade_filter = st.multiselect(
        "Decade",
        options=all_decades,
        default=default_decades,
        key="decade_filter",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="cm-label">⭐ Minimum Rating</div>', unsafe_allow_html=True)
    min_rating = st.slider(
        "Min Rating",
        min_value=0.0, max_value=9.0, value=5.0, step=0.5,
        format="★ %.1f",
        key="min_rating",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="cm-label">Diversity</div>', unsafe_allow_html=True)
    diversify = st.toggle(
        "Diversify results (MMR)",
        value=False,
        key="diversify",
        help="Applies Maximal Marginal Relevance to reduce similar-looking results",
    )

    return {
        "language_codes": language_codes,
        "decade_filter":  decade_filter if decade_filter else all_decades,
        "include_old_movies": include_old_movies,
        "min_rating":     min_rating,
        "diversify":      diversify,
    }


# ── Sidebar Favourites ────────────────────────────────────────────────────────

def render_favourites_sidebar(favourites: list) -> str | None:
    """
    Render the ❤️ My Favourites section.
    Returns the movie_id to remove (or '__CLEAR_ALL__'), or None.
    """
    st.markdown("---")
    count = len(favourites)
    st.markdown(
        f'<div class="cm-label">❤️ MY FAVOURITES &nbsp;<span style="color:#F5C518">{count}</span></div>',
        unsafe_allow_html=True,
    )

    removed_id = None

    if not favourites:
        st.caption("No favourites yet. Click 🤍 on any card to save.")
        return None

    # Show most recently added first (up to 5 preview items)
    for fav in reversed(favourites[-5:]):
        lang_info = LANGUAGE_INFO.get(fav.get("language", ""), {"emoji": "🎬"})
        col_t, col_x = st.columns([5, 1])
        with col_t:
            st.markdown(
                f'<div class="fav-row">{lang_info["emoji"]} '
                f'<strong>{fav["title"]}</strong> ({fav.get("year", "")})</div>',
                unsafe_allow_html=True,
            )
        with col_x:
            st.markdown("<div style='padding-top:2px'>", unsafe_allow_html=True)
            if st.button("✕", key=f"rm_{fav['movie_id']}", help="Remove"):
                removed_id = fav["movie_id"]
            st.markdown("</div>", unsafe_allow_html=True)

    if count > 5:
        st.caption(f"… and {count - 5} more")

    with st.expander(f"View All ({count})"):
        for fav in favourites:
            ra = fav.get("vote_average", 0)
            st.markdown(
                f'<p style="font-size:0.83em;color:#aaa;margin:4px 0">'
                f'🎬 <strong>{fav["title"]}</strong> ({fav.get("year","")}) — ★{ra:.1f}</p>',
                unsafe_allow_html=True,
            )

    csv_data = export_favourites_csv(favourites)
    st.download_button(
        "Export CSV",
        data=csv_data,
        file_name="cinmatch_favourites.csv",
        mime="text/csv",
        use_container_width=True,
        key="export_fav_csv",
    )

    if st.button("Clear All", use_container_width=True, key="clear_all_fav"):
        removed_id = "__CLEAR_ALL__"

    return removed_id


# ── Sidebar Settings ──────────────────────────────────────────────────────────

def render_settings_sidebar() -> dict:
    """Render the Settings expander in sidebar. Returns settings dict."""
    with st.expander("⚙️ Settings"):
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-api03-...",
            key="api_key_input",
            help=(
                "Optional — enables AI-generated 'Why you'll love this' justifications. "
                "Leave blank to use the built-in rule-based fallback."
            ),
        )
        show_score   = st.checkbox("Show match score bar",           value=True,  key="show_score")
        show_just    = st.checkbox("Show 'Why you'll love this'",    value=True,  key="show_just")

    return {
        "api_key":             api_key.strip() or None,
        "show_score_bar":      show_score,
        "show_justifications": show_just,
    }
