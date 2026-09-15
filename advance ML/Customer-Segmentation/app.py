"""
app.py
------
Streamlit dashboard for the Customer Segmentation project.

Two modes:
  1. Manual lookup  - enter Recency/Frequency/Monetary for one customer and see
                      which segment they fall into.
  2. Batch scoring  - upload a raw transactional CSV (same schema as
                      online_retail.csv) and segment every customer in it.

Requires that `models/scaler.pkl`, `models/pca.pkl`, and `models/kmeans.pkl`
already exist (produced by notebooks/Customer_Segmentation.ipynb).
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

import utils

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧭",
    layout="wide",
)

PRIMARY = "#2E5E4E"      # deep evergreen
ACCENT = "#D4A24C"       # warm brass accent
BG_CARD = "#F7F5F0"

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 2rem; }}
        h1, h2, h3 {{ color: {PRIMARY}; }}
        div[data-testid="stMetric"] {{
            background-color: {BG_CARD};
            border-left: 4px solid {ACCENT};
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }}
        .segment-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            background-color: {PRIMARY};
            color: white;
            font-weight: 600;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_models():
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "pca.pkl")
    kmeans_path = os.path.join(MODELS_DIR, "kmeans.pkl")

    missing = [p for p in [scaler_path, pca_path, kmeans_path] if not os.path.exists(p)]
    if missing:
        return None

    scaler = utils.load_model(scaler_path)
    pca = utils.load_model(pca_path)
    kmeans_bundle = utils.load_model(kmeans_path)
    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans_bundle["model"],
        "cluster_to_segment": kmeans_bundle["cluster_to_segment"],
    }


models = load_models()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("🧭 Customer Segmentation Dashboard")
st.caption("RFM feature engineering + KMeans clustering + PCA, trained on the Online Retail dataset.")

if models is None:
    st.error(
        "No trained models found in `models/`. Run "
        "`notebooks/Customer_Segmentation.ipynb` first — it saves "
        "`scaler.pkl`, `pca.pkl`, and `kmeans.pkl` into that folder."
    )
    st.stop()

cluster_to_segment = models["cluster_to_segment"]

tab_lookup, tab_batch, tab_about = st.tabs(["🔎 Single Customer Lookup", "📁 Batch Scoring (CSV Upload)", "ℹ️ About the Segments"])

# --------------------------------------------------------------------------
# Tab 1: Manual single-customer lookup
# --------------------------------------------------------------------------

with tab_lookup:
    st.subheader("Score a single customer")
    st.write("Enter a customer's RFM values to see which segment they'd fall into.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0)

    if st.button("Predict Segment", type="primary"):
        cluster, segment, pca_point = utils.predict_single_customer(
            recency, frequency, monetary,
            models["scaler"], models["pca"], models["kmeans"], cluster_to_segment,
        )

        st.markdown(f"### Predicted segment: <span class='segment-badge'>{segment}</span>", unsafe_allow_html=True)
        st.write(utils.SEGMENT_DESCRIPTIONS.get(segment, "No description available."))

        m1, m2, m3 = st.columns(3)
        m1.metric("Recency", f"{recency} days")
        m2.metric("Frequency", f"{frequency} orders")
        m3.metric("Monetary", f"${monetary:,.2f}")

# --------------------------------------------------------------------------
# Tab 2: Batch scoring via CSV upload
# --------------------------------------------------------------------------

with tab_batch:
    st.subheader("Segment an entire customer base")
    st.write(
        "Upload a raw transactional CSV with the same columns as `online_retail.csv` "
        "(`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
            with st.spinner("Cleaning data, building RFM features, and scoring customers..."):
                result = utils.predict_segments_for_raw_data(
                    raw_df, models["scaler"], models["pca"], models["kmeans"], cluster_to_segment
                )

            st.success(f"Scored {len(result):,} customers.")

            seg_counts = result["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]

            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.dataframe(seg_counts, use_container_width=True, hide_index=True)
            with c2:
                fig_bar = px.bar(
                    seg_counts, x="Segment", y="Customers", color="Segment",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    title="Customers per Segment",
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            fig_scatter = px.scatter(
                result, x="PC1", y="PC2", color="Segment",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Customers in PCA Space, Colored by Segment",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Scored customer table")
            st.dataframe(result, use_container_width=True, hide_index=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored customers as CSV",
                data=csv_bytes,
                file_name="customer_segments.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Couldn't process this file: {e}")

# --------------------------------------------------------------------------
# Tab 3: About the segments
# --------------------------------------------------------------------------

with tab_about:
    st.subheader("What do the segments mean?")
    st.write(
        "Segments are derived by ranking each cluster's average Recency, Frequency, "
        "and Monetary values against the others — clusters with the best combined "
        "rank are named 'Champions', and so on down to 'Hibernating / Lost'."
    )
    for name, desc in utils.SEGMENT_DESCRIPTIONS.items():
        st.markdown(f"**{name}** — {desc}")

    st.subheader("Current cluster → segment mapping")
    mapping_df = pd.DataFrame(
        [{"Cluster": k, "Segment": v} for k, v in cluster_to_segment.items()]
    ).sort_values("Cluster")
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
