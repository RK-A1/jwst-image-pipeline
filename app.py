"""
JWST Explorer — Streamlit app for browsing and analysing JWST pipeline data.

Run from the project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Make include/ importable
sys.path.insert(0, str(Path(__file__).parent / "include"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import get_conn

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGES_DIR = Path(__file__).parent / "include" / "images"

TRAINED_CLASSES = [
    "galaxy", "galaxy cluster", "nebula",
    "star", "solar system", "exoplanet",
]

LABEL_COLORS = {
    "galaxy":                    "#4C72B0",
    "galaxy cluster":            "#DD8452",
    "nebula":                    "#55A868",
    "star":                      "#C44E52",
    "solar system":              "#8172B2",
    "exoplanet":                 "#937860",
    "unclassified":              "#888888",
    "observatory / engineering": "#CCB974",
}

PAGE_SIZE = 30

# ── Styling ───────────────────────────────────────────────────────────────────

_CSS = """
<style>
#MainMenu, footer { visibility: hidden; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
div[data-testid="stMetricLabel"] p {
    color: #64748b;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    color: #0f172a;
    font-size: 1.9rem;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0f172a 0%, #1a2744 100%);
    padding-top: 1.5rem;
}

hr { border-color: #e2e8f0; margin: 0.5rem 0; }
</style>
"""

# Shared Plotly theme applied to every chart
_CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#334155"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def local_image_path(db_path: str) -> str:
    """Remap Docker-internal image_path to host filesystem path."""
    return str(IMAGES_DIR / Path(db_path).name)


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label or "unclassified", "#aaaaaa")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:20px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:.03em;">{label or "—"}</span>'
    )


# ── DB connection (one per Streamlit session) ─────────────────────────────────

@st.cache_resource
def _connection():
    return get_conn(read_only=True)


def db():
    return _connection()


# ── Cached queries ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_overview_metrics():
    con = db()
    total      = con.execute("SELECT count(*) FROM photos").fetchone()[0]
    embedded   = con.execute("SELECT count(*) FROM photos WHERE embedding IS NOT NULL").fetchone()[0]
    labeled    = con.execute("SELECT count(*) FROM photos WHERE canonical_label IS NOT NULL").fetchone()[0]
    predicted  = con.execute("SELECT count(*) FROM photos WHERE predicted_label IS NOT NULL").fetchone()[0]
    return total, embedded, labeled, predicted


@st.cache_data(ttl=300)
def load_label_distributions():
    con = db()
    canonical = con.execute(
        "SELECT coalesce(canonical_label,'(null)') AS label, count(*) AS n "
        "FROM photos GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    predicted = con.execute(
        "SELECT coalesce(predicted_label,'(null)') AS label, count(*) AS n "
        "FROM photos WHERE predicted_label IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    return canonical, predicted


@st.cache_data(ttl=300)
def load_training_runs():
    con = db()
    df = con.execute(
        "SELECT run_id, ts, model_type, accuracy, f1_score, model_path "
        "FROM training_runs ORDER BY ts DESC"
    ).df()
    if not df.empty:
        df["accuracy"] = df["accuracy"].map("{:.4f}".format)
        df["f1_score"] = df["f1_score"].map("{:.4f}".format)
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df.rename(columns={"run_id": "Run ID", "ts": "Time", "model_type": "Type",
                                 "accuracy": "Accuracy", "f1_score": "F1",
                                 "model_path": "Model path"})
    return df


@st.cache_data(ttl=300)
def load_all_photo_titles():
    """Returns list of (photo_id, title) tuples for the selectbox."""
    con = db()
    rows = con.execute(
        "SELECT photo_id, coalesce(title, photo_id) AS title "
        "FROM photos WHERE embedding IS NOT NULL ORDER BY title"
    ).fetchall()
    return rows


@st.cache_data(ttl=300)
def load_confusion_data():
    con = db()
    df = con.execute(
        "SELECT canonical_label, predicted_label, count(*) AS n "
        "FROM photos "
        "WHERE canonical_label IS NOT NULL AND predicted_label IS NOT NULL "
        "GROUP BY 1, 2"
    ).df()
    return df


@st.cache_data(ttl=300)
def load_distinct_labels():
    con = db()
    canonical = sorted([
        r[0] for r in con.execute(
            "SELECT DISTINCT canonical_label FROM photos WHERE canonical_label IS NOT NULL"
        ).fetchall()
    ])
    predicted = sorted([
        r[0] for r in con.execute(
            "SELECT DISTINCT predicted_label FROM photos WHERE predicted_label IS NOT NULL"
        ).fetchall()
    ])
    return canonical, predicted


def load_photos_page(canonical_filter, predicted_filter, page):
    con = db()
    conditions = []
    params = []
    if canonical_filter:
        placeholders = ", ".join("?" * len(canonical_filter))
        conditions.append(f"canonical_label IN ({placeholders})")
        params.extend(canonical_filter)
    if predicted_filter:
        placeholders = ", ".join("?" * len(predicted_filter))
        conditions.append(f"predicted_label IN ({placeholders})")
        params.extend(predicted_filter)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    total = con.execute(f"SELECT count(*) FROM photos {where}", params).fetchone()[0]

    offset = page * PAGE_SIZE
    rows = con.execute(
        f"SELECT photo_id, title, image_path, canonical_label, predicted_label, date_taken "
        f"FROM photos {where} ORDER BY date_taken DESC NULLS LAST "
        f"LIMIT {PAGE_SIZE} OFFSET {offset}",
        params,
    ).fetchall()
    return total, rows


def load_photo_detail(photo_id):
    con = db()
    return con.execute(
        "SELECT photo_id, title, description, tags, image_path, "
        "date_taken, canonical_label, predicted_label "
        "FROM photos WHERE photo_id = ?",
        [photo_id],
    ).fetchone()


def load_similar_photos(photo_id):
    con = db()
    rows = con.execute(
        """
        WITH q AS (SELECT embedding FROM photos WHERE photo_id = ?)
        SELECT p.photo_id, p.title, p.image_path, p.predicted_label, p.canonical_label,
               array_cosine_similarity(p.embedding::FLOAT[2048], q.embedding::FLOAT[2048]) AS similarity
        FROM photos p, q
        WHERE p.photo_id != ?
          AND p.embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 8
        """,
        [photo_id, photo_id],
    ).fetchall()
    return rows


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("Overview")
    st.caption("Pipeline health and label distribution at a glance.")
    st.divider()

    total, embedded, labeled, predicted = load_overview_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Photos", f"{total:,}")
    c2.metric("With Embeddings", f"{embedded:,}")
    c3.metric("Canonical Labels", f"{labeled:,}")
    c4.metric("Predictions", f"{predicted:,}")

    st.divider()

    canonical_df, predicted_df = load_label_distributions()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Canonical label distribution")
        fig = go.Figure(go.Bar(
            x=canonical_df["n"],
            y=canonical_df["label"],
            orientation="h",
            text=canonical_df["n"],
            textposition="outside",
            marker_color=[LABEL_COLORS.get(l, "#aaaaaa") for l in canonical_df["label"]],
            marker_line_width=0,
        ))
        fig.update_layout(
            **_CHART_LAYOUT,
            height=320,
            margin=dict(l=0, r=40, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Predicted label distribution")
        fig2 = go.Figure(go.Bar(
            x=predicted_df["n"],
            y=predicted_df["label"],
            orientation="h",
            text=predicted_df["n"],
            textposition="outside",
            marker_color=[LABEL_COLORS.get(l, "#aaaaaa") for l in predicted_df["label"]],
            marker_line_width=0,
        ))
        fig2.update_layout(
            **_CHART_LAYOUT,
            height=320,
            margin=dict(l=0, r=40, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Training runs")
    runs_df = load_training_runs()
    if runs_df.empty:
        st.info("No training runs recorded yet.")
    else:
        st.dataframe(runs_df, hide_index=True, use_container_width=True)


def page_browser():
    st.title("Photo Browser")
    st.caption("Browse and filter ingested JWST images.")
    st.divider()

    canonical_labels, predicted_labels = load_distinct_labels()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        canon_filter = st.multiselect("Canonical label", canonical_labels)
    with col_f2:
        pred_filter = st.multiselect("Predicted label", predicted_labels)

    # Reset page when filters change
    filter_key = (tuple(canon_filter), tuple(pred_filter))
    if st.session_state.get("_browser_filter_key") != filter_key:
        st.session_state["_browser_filter_key"] = filter_key
        st.session_state["browser_page"] = 0
        st.session_state.pop("selected_photo", None)

    page = st.session_state.get("browser_page", 0)
    total, rows = load_photos_page(canon_filter, pred_filter, page)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    st.caption(f"{total:,} photos  ·  page {page + 1} of {n_pages}")

    # Pagination controls
    p_col1, _, p_col3 = st.columns([1, 6, 1])
    with p_col1:
        if st.button("← Prev", disabled=page == 0, use_container_width=True):
            st.session_state["browser_page"] = page - 1
            st.session_state.pop("selected_photo", None)
            st.rerun()
    with p_col3:
        if st.button("Next →", disabled=page >= n_pages - 1, use_container_width=True):
            st.session_state["browser_page"] = page + 1
            st.session_state.pop("selected_photo", None)
            st.rerun()

    # Image grid
    cols = st.columns(3, gap="medium")
    for i, row in enumerate(rows):
        photo_id, title, image_path, canonical_label, predicted_label, date_taken = row
        path = local_image_path(image_path) if image_path else None
        with cols[i % 3]:
            with st.container(border=True):
                if path and Path(path).exists():
                    st.image(path, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:160px;background:#f1f5f9;border-radius:6px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#94a3b8;font-size:.8rem;">image not found</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"**{(title or photo_id)[:55]}**  \n"
                    f"Predicted: {label_badge(predicted_label)}  \n"
                    f"Canonical: {label_badge(canonical_label)}",
                    unsafe_allow_html=True,
                )
                if st.button("Details", key=f"btn_{photo_id}", use_container_width=True):
                    st.session_state["selected_photo"] = photo_id
                    st.rerun()

    # Detail panel
    selected = st.session_state.get("selected_photo")
    if selected:
        detail = load_photo_detail(selected)
        if detail:
            photo_id, title, description, tags, image_path, date_taken, canonical_label, predicted_label = detail
            st.divider()
            with st.container(border=True):
                d1, d2 = st.columns([1, 2])
                path = local_image_path(image_path) if image_path else None
                with d1:
                    if path and Path(path).exists():
                        st.image(path, use_container_width=True)
                with d2:
                    st.subheader(title or photo_id)
                    st.markdown(
                        f"**Canonical:** {label_badge(canonical_label)} &nbsp; "
                        f"**Predicted:** {label_badge(predicted_label)}",
                        unsafe_allow_html=True,
                    )
                    if date_taken:
                        st.caption(f"Taken: {date_taken}")
                    if description:
                        st.markdown(description[:600] + ("…" if len(description or "") > 600 else ""))
                    if tags:
                        st.markdown("**Tags:** " + ", ".join(tags))
                    if st.button("✕ Close"):
                        st.session_state.pop("selected_photo", None)
                        st.rerun()


def page_similarity():
    st.title("Similarity Search")
    st.caption("Find visually similar photos using ResNet50 embedding cosine similarity.")
    st.divider()

    titles = load_all_photo_titles()
    if not titles:
        st.warning("No embedded photos found. Run the feature extraction DAG first.")
        return

    id_to_title = {r[0]: r[1] for r in titles}
    ids = [r[0] for r in titles]

    selected_id = st.selectbox(
        "Pick a photo",
        options=ids,
        format_func=lambda x: id_to_title.get(x, x),
    )

    if not selected_id:
        return

    detail = load_photo_detail(selected_id)
    if not detail:
        return
    photo_id, title, description, tags, image_path, date_taken, canonical_label, predicted_label = detail

    st.divider()
    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.container(border=True):
            path = local_image_path(image_path) if image_path else None
            if path and Path(path).exists():
                st.image(path, use_container_width=True)
            st.markdown(f"**{title or photo_id}**")
            st.markdown(
                f"Canonical: {label_badge(canonical_label)}  \n"
                f"Predicted: {label_badge(predicted_label)}",
                unsafe_allow_html=True,
            )
            if date_taken:
                st.caption(str(date_taken)[:10])

    with right:
        st.subheader("8 most similar photos")
        similar = load_similar_photos(selected_id)
        if not similar:
            st.info("No similar photos found.")
        else:
            sim_cols = st.columns(4, gap="small")
            for i, row in enumerate(similar):
                s_id, s_title, s_path, s_pred, s_canon, similarity = row
                p = local_image_path(s_path) if s_path else None
                with sim_cols[i % 4]:
                    with st.container(border=True):
                        if p and Path(p).exists():
                            st.image(p, use_container_width=True)
                        st.markdown(
                            f"<small>**{(s_title or s_id)[:35]}**</small>  \n"
                            f"{label_badge(s_pred)}  \n"
                            f"<small style='color:#64748b'>sim: {similarity:.3f}</small>",
                            unsafe_allow_html=True,
                        )


def page_performance():
    st.title("Model Performance")
    st.caption("Per-class accuracy and confusion matrix across canonical and predicted labels.")
    st.divider()

    confusion_df = load_confusion_data()
    if confusion_df.empty:
        st.info("No predictions yet. Run the ingest DAG to generate predictions.")
        return

    # Build pivot restricted to trained classes (rows) × all predicted (cols)
    predicted_cols = sorted(confusion_df["predicted_label"].unique().tolist())
    pivot = (
        confusion_df[confusion_df["canonical_label"].isin(TRAINED_CLASSES)]
        .pivot_table(index="canonical_label", columns="predicted_label",
                     values="n", aggfunc="sum", fill_value=0)
        .reindex(index=TRAINED_CLASSES, fill_value=0)
        .reindex(columns=predicted_cols, fill_value=0)
    )

    # Per-class accuracy
    st.subheader("Per-class accuracy")
    acc_data = []
    for cls in TRAINED_CLASSES:
        if cls not in pivot.index:
            continue
        row = pivot.loc[cls]
        total = row.sum()
        correct = row.get(cls, 0)
        if total > 0:
            acc_data.append({"Class": cls, "Accuracy": correct / total, "n": int(total)})

    if acc_data:
        acc_df = pd.DataFrame(acc_data).sort_values("Accuracy", ascending=True)
        fig = go.Figure(go.Bar(
            x=acc_df["Accuracy"],
            y=acc_df["Class"],
            orientation="h",
            text=[f"{a:.1%}  (n={n})" for a, n in zip(acc_df["Accuracy"], acc_df["n"])],
            textposition="outside",
            marker_color=[LABEL_COLORS.get(c, "#aaaaaa") for c in acc_df["Class"]],
            marker_line_width=0,
        ))
        fig.update_layout(
            **_CHART_LAYOUT,
            xaxis=dict(range=[0, 1.2], tickformat=".0%", showgrid=False),
            yaxis=dict(showgrid=False),
            height=300,
            margin=dict(l=0, r=80, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Confusion matrix")
    z = pivot.values
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)

    heatmap = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=z, texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    heatmap.update_layout(
        **_CHART_LAYOUT,
        xaxis_title="Predicted",
        yaxis_title="Canonical",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.divider()
    st.subheader("Training runs")
    runs_df = load_training_runs()
    if runs_df.empty:
        st.info("No training runs recorded yet.")
    else:
        st.dataframe(runs_df, hide_index=True, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="JWST Explorer", page_icon="🔭", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding-bottom:1rem;'>"
        "<span style='font-size:2.4rem;'>🔭</span><br>"
        "<span style='color:#f1f5f9;font-size:1.1rem;font-weight:700;'>JWST Explorer</span><br>"
        "<span style='color:#64748b;font-size:.75rem;'>James Webb Space Telescope</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e3a5f;margin:0 0 1rem 0'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Photo Browser", "Similarity Search", "Model Performance"],
        label_visibility="collapsed",
    )

if page == "Overview":
    page_overview()
elif page == "Photo Browser":
    page_browser()
elif page == "Similarity Search":
    page_similarity()
elif page == "Model Performance":
    page_performance()
