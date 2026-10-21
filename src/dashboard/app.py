
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie ML Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = (
    "https://ebayw333oc.execute-api.eu-north-1.amazonaws.com/default"
)

API_TIMEOUT = 30

# Previously recorded evaluation metrics.
# These values are not recalculated live by this dashboard.
MODEL_RESULTS = {
    "SVD": {
        "RMSE": 0.9365,
        "MAE": 0.7358,
    },
    "Neural CF": {
        "RMSE": 0.9436,
        "MAE": 0.7521,
    },
}

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #151a24;
        border-right: 1px solid #283044;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.25rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    .eyebrow {
        color: #75a7ff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .subtitle {
        color: #a9b2c5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .movie-card {
        background: #171e2b;
        border: 1px solid #2b3548;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .movie-rank {
        color: #82aaff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .movie-title {
        color: #f1f5fb;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .movie-rating {
        color: #f5c76b;
        font-size: 1rem;
        font-weight: 700;
    }

    .small-muted {
        color: #a9b2c5;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: #171e2b;
        border: 1px solid #2b3548;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9b2c5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def get_field(item, candidates, default=None):
    """Return the first available field from a dictionary."""
    if not isinstance(item, dict):
        return default

    for key in candidates:
        if key in item and item[key] is not None:
            return item[key]

    return default


def normalize_recommendations(payload):
    """Normalize common recommendation API response formats."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("recommendations", [])
    else:
        items = []

    if not isinstance(items, list):
        return []

    normalized = []

    for item in items:
        if not isinstance(item, dict):
            continue

        movie_id = get_field(
            item,
            ["movie_id", "movieId", "id", "item_id"],
            default=None,
        )

        title = get_field(
            item,
            ["title", "movie_title", "movie", "name"],
            default="Unknown movie",
        )

        rating = get_field(
            item,
            [
                "predicted_rating",
                "predicted_rating_score",
                "estimated_rating",
                "rating",
                "score",
                "prediction",
            ],
            default=None,
        )

        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None

        try:
            movie_id = int(movie_id) if movie_id is not None else None
        except (TypeError, ValueError):
            pass

        normalized.append(
            {
                "movie_id": movie_id,
                "title": str(title),
                "rating": rating,
            }
        )

    return normalized


def fetch_recommendations(user_id, number_of_movies):
    """Call the deployed AWS API and return recommendations."""
    endpoint = f"{API_URL}/recommendations/{int(user_id)}"

    try:
        response = requests.get(
            endpoint,
            params={"n": int(number_of_movies)},
            timeout=API_TIMEOUT,
        )

        if response.status_code == 404:
            return None, "User not found in the recommendation dataset."

        if response.status_code == 400:
            return None, "Invalid request. Check the user ID and movie count."

        response.raise_for_status()

        payload = response.json()
        recommendations = normalize_recommendations(payload)

        return {
            "payload": payload,
            "recommendations": recommendations,
        }, None

    except requests.Timeout:
        return None, "The AWS API request timed out. Please try again."

    except requests.RequestException as exc:
        return None, f"Unable to reach the AWS API: {exc}"

    except ValueError:
        return None, "The AWS API returned an invalid JSON response."


def check_api_status():
    """Check API availability using a real recommendation request."""
    try:
        response = requests.get(
            f"{API_URL}/recommendations/1",
            params={"n": 1},
            timeout=10,
        )
        return response.ok, (
            "AWS API is responding successfully"
            if response.ok
            else f"API returned HTTP {response.status_code}"
        )

    except requests.RequestException:
        return False, "Unable to reach AWS API"


def render_movie_cards(recommendations):
    """Render recommendations as individual movie cards."""
    if not recommendations:
        st.warning(
            "No recommendations were returned. "
            "Check the API response below."
        )
        return

    for rank, movie in enumerate(recommendations, start=1):
        movie_id = movie["movie_id"]
        title = movie["title"]
        rating = movie["rating"]

        id_text = (
            f"MOVIE ID {movie_id}"
            if movie_id is not None
            else "MOVIE ID unavailable"
        )

        rating_text = (
            f"{rating:.2f} / 5"
            if rating is not None
            else "Predicted rating unavailable"
        )

        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-rank">
                    TOP #{rank} &nbsp; | &nbsp; {id_text}
                </div>
                <div class="movie-title">
                    {title}
                </div>
                <div class="movie-rating">
                    Rating: {rating_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header():
    st.markdown(
        '<div class="eyebrow">'
        'PERSONALIZED DISCOVERY | MACHINE LEARNING'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">Movie Recommender</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover your next favorite film with personalized '
        'recommendations powered by collaborative filtering.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.divider()
    st.caption(
        "MOVIE RECOMMENDER | ML ENGINEERING PROJECT | "
        "SVD | ALS | NEURAL COLLABORATIVE FILTERING"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎬 Movie ML Studio")
    st.caption("MOVIE DISCOVERY PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Discover Movies",
            "Model Analytics",
            "System Status",
        ],
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
        step=1,
    )

    st.caption("Powered by AWS Lambda | API Gateway | S3")


# ============================================================
# HEADER AND SUMMARY METRICS
# ============================================================

render_header()

metric_columns = st.columns(4)

metric_columns[0].metric("REGISTERED USERS", "943")
metric_columns[1].metric("MOVIES", "1,682")
metric_columns[2].metric("USER RATINGS", "100,000")
metric_columns[3].metric("ML MODELS", "3")

st.divider()


# ============================================================
# PAGE 1: DISCOVER MOVIES
# ============================================================

if page == "Discover Movies":

    st.markdown(
        '<div class="section-title">Your recommendations</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Personalized recommendations for user {int(user_id)}."
    )

    if st.button(
        "Generate recommendations",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Requesting recommendations from AWS..."):
            result, error = fetch_recommendations(
                user_id,
                recommendation_count,
            )

        if error:
            st.session_state.pop("movie_payload", None)
            st.session_state.pop("movie_user_id", None)
            st.error(error)
        else:
            st.session_state["movie_payload"] = result["payload"]
            st.session_state["movie_recommendations"] = (
                result["recommendations"]
            )
            st.session_state["movie_user_id"] = int(user_id)
            st.session_state["movie_count"] = recommendation_count

    saved_payload = st.session_state.get("movie_payload")
    saved_user_id = st.session_state.get("movie_user_id")

    if saved_payload is not None and saved_user_id == int(user_id):
        recommendations = st.session_state.get(
            "movie_recommendations",
            [],
        )

        info_columns = st.columns(3)

        info_columns[0].metric(
            "Selected user",
            str(saved_user_id),
        )

        info_columns[1].metric(
            "Movies returned",
            str(len(recommendations)),
        )

        info_columns[2].metric(
            "Recommendation source",
            str(
                saved_payload.get("source", "SVD")
                if isinstance(saved_payload, dict)
                else "SVD"
            ),
        )

        render_movie_cards(recommendations)

        with st.expander("View raw API response"):
            st.json(saved_payload)

    else:
        st.info(
            "Choose a user and the number of movies, "
            "then click Generate recommendations."
        )

    st.divider()

    st.markdown("### How it works")

    info = st.columns(3)

    info[0].markdown(
        "**01 | User profile**\n\n"
        "Select a MovieLens user."
    )

    info[1].markdown(
        "**02 | Recommendation engine**\n\n"
        "AWS Lambda runs the SVD model to generate "
        "personalized predictions."
    )

    info[2].markdown(
        "**03 | Ranked results**\n\n"
        "Explore movies ordered by predicted rating."
    )


# ============================================================
# PAGE 2: MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":

    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Previously measured held-out evaluation results. "
        "Lower RMSE and MAE indicate smaller prediction errors."
    )

    results_df = pd.DataFrame.from_dict(
        MODEL_RESULTS,
        orient="index",
    )

    results_df.index.name = "Model"
    results_df = results_df.reset_index()

    # RMSE and MAE comparison charts
    left, right = st.columns(2)

    with left:
        st.markdown("#### RMSE comparison")

        rmse_fig = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            color="Model",
            text="RMSE",
            title="Root Mean Squared Error",
        )

        rmse_fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
        )

        rmse_fig.update_layout(
            showlegend=False,
            yaxis_title="RMSE (lower is better)",
            xaxis_title="",
            margin=dict(l=10, r=10, t=60, b=10),
        )

        st.plotly_chart(
            rmse_fig,
            use_container_width=True,
        )

    with right:
        st.markdown("#### MAE comparison")

        mae_fig = px.bar(
            results_df,
            x="Model",
            y="MAE",
            color="Model",
            text="MAE",
            title="Mean Absolute Error",
        )

        mae_fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
        )

        mae_fig.update_layout(
            showlegend=False,
            yaxis_title="MAE (lower is better)",
            xaxis_title="",
            margin=dict(l=10, r=10, t=60, b=10),
        )

        st.plotly_chart(
            mae_fig,
            use_container_width=True,
        )

    # Evaluation results table
    st.markdown("#### Evaluation results")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    # Model registry
    st.markdown("#### Model registry")

    registry = pd.DataFrame(
        [
            {
                "Model": "SVD",
                "Artifact": "svd_model.pkl",
                "Evaluation": "Previously evaluated",
            },
            {
                "Model": "ALS",
                "Artifact": "als_model.npz",
                "Evaluation": (
                    "Artifact available; comparable test metrics "
                    "not verified"
                ),
            },
            {
                "Model": "Neural CF",
                "Artifact": "neural_cf.pt",
                "Evaluation": "Previously evaluated",
            },
        ]
    )

    st.dataframe(
        registry,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "SVD: RMSE 0.9365, MAE 0.7358. "
        "Neural CF: RMSE 0.9436, MAE 0.7521. "
        "These are previously recorded results, not a live evaluation."
    )


# ============================================================
# PAGE 3: SYSTEM STATUS
# ============================================================

elif page == "System Status":

    st.markdown(
        '<div class="section-title">System status</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Checking AWS API availability..."):
        api_online, api_message = check_api_status()

    st.markdown("#### AWS Lambda API")

    if api_online:
        st.success(api_message)
    else:
        st.error(api_message)

    st.markdown("#### API endpoint")

    st.code(
        f"{API_URL}/recommendations/{{user_id}}?n=10",
        language="text",
    )

    st.markdown("#### Project architecture")

    st.markdown(
        """
        **Data and model training**

        `MovieLens 100K` → `PostgreSQL` → `SVD / ALS / Neural CF`

        `Airflow` and `MLflow` support the training and experiment workflow.

        **Online recommendation API**

        `Streamlit dashboard` → `Amazon API Gateway` → `AWS Lambda (SVD)`

        `AWS Lambda` loads the model package and retrieves movie and
        rating data from `Amazon S3`.

        **Local services**

        `FastAPI` and `Redis` are available as part of the local
        application architecture.
        """
    )

    st.markdown("#### Deployment details")

    deployment_df = pd.DataFrame(
        [
            {
                "Component": "Streamlit",
                "Role": "Interactive dashboard",
                "Environment": "Local",
            },
            {
                "Component": "Amazon API Gateway",
                "Role": "HTTP API endpoint",
                "Environment": "AWS",
            },
            {
                "Component": "AWS Lambda",
                "Role": "SVD inference",
                "Environment": "AWS",
            },
            {
                "Component": "Amazon S3",
                "Role": "Model and data storage",
                "Environment": "AWS",
            },
            {
                "Component": "FastAPI / Redis",
                "Role": "Local API and cache",
                "Environment": "Local architecture",
            },
            {
                "Component": "Airflow / MLflow",
                "Role": "Training workflow and experiment tracking",
                "Environment": "Project services",
            },
        ]
    )

    st.dataframe(
        deployment_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

render_footer()
