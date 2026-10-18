import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Movie Recommender | ML Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Previously measured evaluation results from this project.
# ALS remains unevaluated here until a comparable test result is available.
MODEL_RESULTS = {
    "SVD": {"RMSE": 0.9365, "MAE": 0.7358},
    "Neural CF": {"RMSE": 0.9436, "MAE": 0.7521},
}

st.markdown("""
<style>
.stApp {
    background: #0b0e19;
    color: #f2f3fa;
}
[data-testid="stSidebar"] {
    background: #111629;
    border-right: 1px solid #252b43;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #191f3a, #30204b 70%, #171a30);
    border: 1px solid #383354;
    margin-bottom: 24px;
}
.hero-label {
    color: #b9a8ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}
.hero h1 {
    color: #ffffff;
    font-size: 38px;
    margin: 8px 0;
}
.hero p {
    color: #c4c8dc;
    margin-bottom: 0;
}
.metric-card {
    background: #141a2b;
    border: 1px solid #292f49;
    padding: 18px;
    border-radius: 15px;
    min-height: 105px;
}
.metric-label {
    color: #aab1ca;
    font-size: 13px;
}
.metric-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 750;
    margin-top: 6px;
}
.movie-card {
    background: linear-gradient(145deg, #171d31, #121625);
    border: 1px solid #303650;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
    min-height: 142px;
}
.movie-rank {
    color: #b8a5ff;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 700;
}
.movie-title {
    color: #ffffff;
    font-size: 17px;
    font-weight: 650;
    margin: 12px 0;
}
.movie-rating {
    color: #f4d58b;
    font-size: 16px;
    font-weight: 700;
}
.section-title {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
    margin-top: 12px;
}
.small-muted {
    color: #aab1ca;
    font-size: 13px;
}
div[data-testid="stMetric"] {
    background: #141a2b;
    border: 1px solid #292f49;
    padding: 14px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #aab1ca;
}
div[data-testid="stMetricValue"] {
    color: #ffffff;
}
.stButton button[kind="primary"] {
    background: #7961e8;
    border: 0;
    border-radius: 10px;
    color: white;
    font-weight: 650;
}
hr {
    border-color: #292f49;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def get_title(movie):
    return str(movie.get("title", f"Movie {movie.get('movie_id', '?')}"))


def get_rating(movie):
    value = movie.get("predicted_rating")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Sidebar
with st.sidebar:
    st.markdown("## 🎬 ML Studio")
    st.caption("MOVIE DISCOVERY PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Discover Movies", "Model Analytics", "System Status"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("#### Recommendation settings")
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=943,
        value=1,
        step=1,
    )
    recommendation_count = st.slider(
        "Number of movies",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )

    fetch_clicked = st.button(
        "✨ Generate recommendations",
        type="primary",
        use_container_width=True,
    )

    if fetch_clicked:
        try:
            response = requests.get(
                f"{API_URL}/recommendations/{int(user_id)}",
                params={"n": recommendation_count},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["movie_payload"] = response.json()
            st.session_state["movie_payload_user"] = int(user_id)
            st.session_state["movie_payload_error"] = None
        except requests.RequestException as exc:
            st.session_state["movie_payload_error"] = str(exc)

    st.divider()
    st.caption("Powered by FastAPI · PostgreSQL · Redis")


# Header
st.markdown("""
<div class="hero">
    <div class="hero-label">PERSONALIZED DISCOVERY · MACHINE LEARNING</div>
    <h1>Movie Recommender</h1>
    <p>Discover your next favorite film with personalized recommendations
    powered by collaborative filtering.</p>
</div>
""", unsafe_allow_html=True)

# Dataset overview
overview = st.columns(4)
with overview[0]:
    metric_card("REGISTERED USERS", "943")
with overview[1]:
    metric_card("MOVIES", "1,682")
with overview[2]:
    metric_card("USER RATINGS", "100,000")
with overview[3]:
    metric_card("ML MODELS", "3")

st.write("")

# API health indicator
api_online = False
api_message = "API is unreachable"
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    api_online = health.ok
    api_message = "API online" if health.ok else f"HTTP {health.status_code}"
except requests.RequestException:
    pass

if page == "Discover Movies":
    st.markdown(
        '<div class="section-title">🍿 Your recommendations</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Recommendations for user {int(user_id)} · "
        "Sorted by predicted rating"
    )

    error = st.session_state.get("movie_payload_error")
    payload = st.session_state.get("movie_payload")
    payload_user = st.session_state.get("movie_payload_user")

    if error:
        st.error(f"Could not contact the recommendation API: {error}")
        st.info("Check that FastAPI is running on port 8000.")
    elif payload and payload_user == int(user_id):
        movies = payload.get("recommendations", [])
        source = payload.get("source", "unknown")

        summary = st.columns(3)
        summary[0].metric("Selected user", int(user_id))
        summary[1].metric("Movies returned", len(movies))
        summary[2].metric("Recommendation source", str(source).upper())

        if not movies:
            st.info("The API returned no recommendations.")
        else:
            for start in range(0, len(movies), 2):
                cols = st.columns(2)
                for offset, col in enumerate(cols):
                    index = start + offset
                    if index >= len(movies):
                        continue

                    movie = movies[index]
                    title = get_title(movie)
                    rating = get_rating(movie)
                    rating_text = (
                        f"★ {rating:.2f} / 5"
                        if rating is not None
                        else "Rating unavailable"
                    )
                    movie_id = movie.get("movie_id", "N/A")

                    with col:
                        st.markdown(
                            f'<div class="movie-card">'
                            f'<div class="movie-rank">TOP #{index + 1}'
                            f' &nbsp; · &nbsp; MOVIE ID {movie_id}</div>'
                            f'<div class="movie-title">{title}</div>'
                            f'<div class="movie-rating">{rating_text}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        with st.expander("View raw API response"):
            st.json(payload)
    else:
        st.info(
            "Choose a user and click **Generate recommendations** "
            "in the sidebar to discover movies."
        )

    st.divider()
    st.markdown("#### How it works")
    info = st.columns(3)
    info[0].markdown("**01 · User profile**\n\nSelect a MovieLens user.")
    info[1].markdown("**02 · Recommendation engine**\n\nFastAPI retrieves personalized predictions.")
    info[2].markdown("**03 · Ranked results**\n\nExplore movies ordered by predicted rating.")

elif page == "Model Analytics":
    st.markdown(
        '<div class="section-title">📊 Model performance</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Previously measured held-out evaluation results. "
        "Lower RMSE and MAE indicate smaller prediction errors."
    )

    results_df = pd.DataFrame(MODEL_RESULTS).T
    results_df.index.name = "Model"
    results_df = results_df.reset_index()

    left, right = st.columns(2)
    with left:
        st.markdown("#### RMSE comparison")
        st.bar_chart(results_df.set_index("Model")[["RMSE"]])
    with right:
        st.markdown("#### MAE comparison")
        st.bar_chart(results_df.set_index("Model")[["MAE"]])

    st.markdown("#### Evaluation results")
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Model registry")
    registry = pd.DataFrame([
        {
            "Model": "SVD",
            "Artifact": "svd_model.pkl",
            "Evaluation": "Available",
        },
        {
            "Model": "ALS",
            "Artifact": "als_model.npz",
            "Evaluation": "Not shown — comparable test metrics not verified",
        },
        {
            "Model": "Neural CF",
            "Artifact": "neural_cf.pt",
            "Evaluation": "Available",
        },
    ])
    st.dataframe(registry, use_container_width=True, hide_index=True)

    st.caption(
        "SVD: RMSE 0.9365, MAE 0.7358. "
        "Neural CF: RMSE 0.9436, MAE 0.7521. "
        "These are previously recorded results, not a live evaluation."
    )

elif page == "System Status":
    st.markdown(
        '<div class="section-title">🛠️ System status</div>',
        unsafe_allow_html=True,
    )

    status_col, url_col = st.columns(2)
    with status_col:
        st.markdown("#### FastAPI")
        if api_online:
            st.success("Healthy — /health responded successfully")
        else:
            st.error(api_message)

    with url_col:
        st.markdown("#### API endpoint")
        st.code(f"{API_URL}/recommendations/{{user_id}}?n=10")

    st.markdown("#### Project architecture")
    st.markdown("""
    `MovieLens 100K` → `PostgreSQL` → `SVD / ALS / Neural CF`

    `FastAPI` → `Redis cache` → `Streamlit dashboard`

    `Airflow` and `MLflow` support the training and experiment workflow.
    """)

st.divider()
st.markdown(
    '<div class="small-muted">'
    'MOVIE RECOMMENDER · ML ENGINEERING PROJECT · '
    'SVD · ALS · NEURAL COLLABORATIVE FILTERING'
    '</div>',
    unsafe_allow_html=True,
)
