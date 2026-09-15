"""
app.py
------
Interactive Streamlit dashboard for the Customer Segmentation project.

Tab 1 "Upload & Cluster (Live)" — the main wizard: upload any CSV, the app
auto-detects Online-Retail-style transactional data and builds RFM features,
or (if it doesn't look transactional, or you choose to override) lets you
manually pick which columns to use as features. Then choose an algorithm
(K-Means / Hierarchical / DBSCAN), set its parameters, run it live, and view
results.

Tab 2 "Quick Lookup (Saved Model)" — reuses a K-Means pipeline previously
saved either from this app (Tab 1's "save" button) or from
notebooks/Customer_Segmentation.ipynb, to score a single new record or a new
batch. Only K-Means models can be saved for this, because scikit-learn's
Hierarchical/DBSCAN implementations can't score unseen data (see utils.py).

Tab 3 "About" — explains the algorithms and segment naming.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

import utils

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="Customer Segmentation", layout="wide")

PRIMARY = "#2E5E4E"
ACCENT = "#D4A24C"
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
        .caveat-box {{
            background-color: #FFF6E5;
            border-left: 4px solid {ACCENT};
            padding: 0.6rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.75rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

st.title("Customer Segmentation Dashboard")
st.caption("Upload any customer CSV, choose your features, pick an algorithm, and cluster your customers.")

tab_live, tab_lookup, tab_about = st.tabs(
    ["Upload & Cluster (Live)", "Quick Lookup (Saved Model)", "About"]
)

# ==========================================================================
# TAB 1 — Upload & Cluster (Live)
# ==========================================================================

with tab_live:
    st.subheader("Step 1 — Upload your CSV")
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"], key="live_upload")

    if uploaded is None:
        st.info("Upload a CSV to get started. Retail transactional data (InvoiceNo, InvoiceDate, "
                "Quantity, UnitPrice, CustomerID, ...) will automatically get RFM features built for it. "
                "Any other CSV lets you pick which columns to cluster on.")
    else:
        try:
            raw_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            st.stop()

        st.write(f"Loaded **{len(raw_df):,} rows**, **{len(raw_df.columns)} columns**.")
        with st.expander("Preview raw data"):
            st.dataframe(raw_df.head(20), use_container_width=True)

        is_retail = utils.detect_retail_schema(raw_df)

        st.subheader("Step 2 — Build features")
        if is_retail:
            st.success("Detected Online-Retail-style transactional columns — RFM features "
                       "(Recency, Frequency, Monetary) will be computed automatically.")
        else:
            st.info("Didn't detect retail transactional columns. Pick which columns to use as "
                    "clustering features below.")

        use_manual = is_retail and st.checkbox(
            "Use manual column selection instead of auto-RFM", value=False,
            help="Override auto-detected RFM and pick your own feature columns from this file."
        )

        mode = "generic" if (use_manual or not is_retail) else "rfm"

        feature_df = None       # scaled/transformed features ready for clustering
        reference_df = None     # human-readable table to attach cluster/segment onto
        numeric_cols, categorical_cols = [], []
        preprocessor_or_scaler = None

        if mode == "rfm":
            log_rfm, rfm_indexed = utils.build_rfm_features(raw_df)
            scaled_df, scaler = utils.scale_features(log_rfm, fit=True)
            feature_df = scaled_df
            reference_df = rfm_indexed
            preprocessor_or_scaler = scaler
            st.dataframe(rfm_indexed.head(10), use_container_width=True)
        else:
            all_cols = list(raw_df.columns)
            default_numeric = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_df[c])]
            default_categorical = [c for c in all_cols if c not in default_numeric]

            col1, col2 = st.columns(2)
            with col1:
                numeric_cols = st.multiselect("Numeric feature columns", options=all_cols, default=default_numeric[:5])
            with col2:
                categorical_cols = st.multiselect("Categorical feature columns", options=all_cols, default=default_categorical[:3])

            if not numeric_cols and not categorical_cols:
                st.warning("Pick at least one numeric or categorical column to continue.")
                st.stop()

            cleaned = utils.clean_generic_data(raw_df)
            transformed_df, preprocessor = utils.preprocess_features(cleaned, numeric_cols, categorical_cols, fit=True)
            feature_df = transformed_df
            reference_df = cleaned
            preprocessor_or_scaler = preprocessor
            st.write(f"Built **{transformed_df.shape[1]} features** from your selected columns (after imputing/scaling/encoding).")

        st.subheader("Step 3 — Choose an algorithm")
        algorithm = st.radio(
            "Clustering algorithm",
            options=utils.ALGORITHMS,
            format_func=lambda a: utils.ALGORITHM_LABELS[a],
            horizontal=True,
        )

        if algorithm != "kmeans":
            st.markdown(
                f'<div class="caveat-box"> <b>{utils.ALGORITHM_LABELS[algorithm]}</b> can\'t score '
                "brand-new data later — every time you want cluster labels, you re-run it on the exact "
                "dataset you want labeled. Only K-Means models can be saved for the Quick Lookup tab.</div>",
                unsafe_allow_html=True,
            )

        n_rows = len(feature_df)
        if algorithm == "hierarchical" and n_rows > 5000:
            st.warning(
                f"Hierarchical clustering is memory/time-heavy for {n_rows:,} rows. "
                "Consider sampling below."
            )
            sample_it = st.checkbox("Sample down to 5,000 rows before clustering", value=True)
            if sample_it:
                feature_df = feature_df.sample(5000, random_state=42)
                reference_df = reference_df.loc[feature_df.index]

        st.subheader("Step 4 — Set algorithm parameters")

        params = {}
        if algorithm == "kmeans":
            params["n_clusters"] = st.slider("Number of clusters (k)", 2, 10, 4)
            with st.expander("Help me choose k (elbow + silhouette)"):
                if st.button("Compute diagnostics", key="kmeans_diag"):
                    with st.spinner("Fitting KMeans across k = 2..9..."):
                        k_range, inertias, silhouettes = utils.elbow_and_silhouette(feature_df)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(px.line(x=k_range, y=inertias, markers=True,
                                                 labels={"x": "k", "y": "Inertia"}, title="Elbow Method"),
                                         use_container_width=True)
                    with c2:
                        st.plotly_chart(px.line(x=k_range, y=silhouettes, markers=True,
                                                 labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Score"),
                                         use_container_width=True)

        elif algorithm == "hierarchical":
            params["n_clusters"] = st.slider("Number of clusters", 2, 10, 4)
            params["linkage"] = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            with st.expander("Help me choose (elbow + silhouette, via a quick KMeans proxy)"):
                if st.button("Compute diagnostics", key="hier_diag"):
                    with st.spinner("Computing diagnostics..."):
                        k_range, inertias, silhouettes = utils.elbow_and_silhouette(feature_df)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(px.line(x=k_range, y=inertias, markers=True,
                                                 labels={"x": "k", "y": "Inertia"}, title="Elbow Method (KMeans proxy)"),
                                         use_container_width=True)
                    with c2:
                        st.plotly_chart(px.line(x=k_range, y=silhouettes, markers=True,
                                                 labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Score"),
                                         use_container_width=True)

        elif algorithm == "dbscan":
            col1, col2 = st.columns(2)
            with col1:
                params["eps"] = st.number_input("eps (neighborhood radius)", min_value=0.01, value=0.5, step=0.05)
            with col2:
                params["min_samples"] = st.slider("min_samples", 2, 20, 5)
            with st.expander("Help me choose eps (k-distance plot)"):
                if st.button("Compute k-distance graph", key="dbscan_diag"):
                    with st.spinner("Computing nearest-neighbor distances..."):
                        k_dist = utils.compute_k_distance(feature_df, k=params["min_samples"])
                    st.plotly_chart(
                        px.line(y=k_dist, labels={"index": "Points (sorted)", "y": f"Distance to {params['min_samples']}-th neighbor"},
                                title="K-Distance Graph — look for the 'elbow'"),
                        use_container_width=True,
                    )
                    st.caption("Pick `eps` around the y-value where the curve bends sharply upward.")

        st.subheader("Step 5 — Run clustering")
        if st.button("Run Clustering", type="primary"):
            with st.spinner(f"Running {utils.ALGORITHM_LABELS[algorithm]}..."):
                model, labels, supports_predict = utils.run_clustering(algorithm, feature_df, **params)
                pca_df, pca = utils.apply_pca(feature_df, fit=True, n_components=2)

                result_ref = reference_df.copy()
                result_ref["Cluster"] = labels

                if mode == "rfm":
                    cluster_to_segment = utils.label_segments(result_ref, cluster_col="Cluster")
                else:
                    cluster_to_segment, profile = utils.label_segments_generic(
                        result_ref, numeric_cols, categorical_cols, cluster_col="Cluster"
                    )

                result_ref["Segment"] = result_ref["Cluster"].map(cluster_to_segment)
                result_ref["PC1"] = pca_df["PC1"].values
                result_ref["PC2"] = pca_df["PC2"].values

            st.session_state["live_result"] = result_ref
            st.session_state["live_mode"] = mode
            st.session_state["live_algorithm"] = algorithm
            st.session_state["live_supports_predict"] = supports_predict
            st.session_state["live_model"] = model
            st.session_state["live_preprocessor"] = preprocessor_or_scaler
            st.session_state["live_pca"] = pca
            st.session_state["live_cluster_to_segment"] = cluster_to_segment
            st.session_state["live_numeric_cols"] = numeric_cols
            st.session_state["live_categorical_cols"] = categorical_cols

        if "live_result" in st.session_state:
            result = st.session_state["live_result"]
            algo_used = st.session_state["live_algorithm"]

            st.subheader("Results")
            n_clusters_found = result["Cluster"].nunique()
            m1, m2, m3 = st.columns(3)
            m1.metric("Algorithm", utils.ALGORITHM_LABELS[algo_used])
            m2.metric("Clusters found", n_clusters_found)
            if -1 in result["Cluster"].unique():
                m3.metric("Noise points", int((result["Cluster"] == -1).sum()))
            else:
                m3.metric("Total records", len(result))

            seg_counts = result["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]

            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.dataframe(seg_counts, use_container_width=True, hide_index=True)
            with c2:
                fig_bar = px.bar(seg_counts, x="Segment", y="Count", color="Segment",
                                  color_discrete_sequence=px.colors.qualitative.Safe, title="Customers per Segment")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            fig_scatter = px.scatter(
                result, x="PC1", y="PC2", color="Segment",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Customers in PCA Space, Colored by Segment",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.dataframe(result, use_container_width=True)

            csv_bytes = result.to_csv(index=True).encode("utf-8")
            st.download_button("Download clustered data as CSV", data=csv_bytes,
                                file_name="customer_segments.csv", mime="text/csv")

            st.subheader("Step 6 — Save this pipeline (optional)")
            if st.session_state["live_supports_predict"]:
                st.write("This is a K-Means model, so it can score new customers later. "
                         "Save it to reuse in the **Quick Lookup** tab.")
                if st.button("Save as the active model"):
                    utils.save_model(st.session_state["live_preprocessor"], os.path.join(MODELS_DIR, "scaler.pkl"))
                    utils.save_model(st.session_state["live_pca"], os.path.join(MODELS_DIR, "pca.pkl"))
                    bundle = {
                        "algorithm": algo_used,
                        "mode": st.session_state["live_mode"],
                        "model": st.session_state["live_model"],
                        "cluster_to_segment": st.session_state["live_cluster_to_segment"],
                        "numeric_cols": st.session_state["live_numeric_cols"],
                        "categorical_cols": st.session_state["live_categorical_cols"],
                    }
                    utils.save_model(bundle, os.path.join(MODELS_DIR, "cluster_model.pkl"))
                    st.success("Saved! Head to the Quick Lookup tab to use it.")
            else:
                st.info(f"{utils.ALGORITHM_LABELS[algo_used]} can't score new/unseen data, so there's "
                        "nothing useful to save for later lookups — re-run this tab whenever you have new data.")

# ==========================================================================
# TAB 2 — Quick Lookup (Saved Model)
# ==========================================================================

with tab_lookup:
    st.subheader("Score new data with a saved K-Means model")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "pca.pkl")
    bundle_path = os.path.join(MODELS_DIR, "cluster_model.pkl")

    missing = [p for p in [scaler_path, pca_path, bundle_path] if not os.path.exists(p)]

    if missing:
        st.warning(
            "No saved model found yet. Either run the training notebook "
            "(`notebooks/Customer_Segmentation.ipynb`), or go to the **Upload & Cluster (Live)** "
            "tab, run K-Means, and click **Save as the active model**."
        )
    else:
        preprocessor = utils.load_model(scaler_path)
        pca = utils.load_model(pca_path)
        bundle = utils.load_model(bundle_path)

        if bundle["algorithm"] != "kmeans":
            st.error(
                f"The currently saved model is **{utils.ALGORITHM_LABELS.get(bundle['algorithm'], bundle['algorithm'])}**, "
                "which can't score new data. Go save a K-Means model from the Upload & Cluster tab instead."
            )
        else:
            kmeans = bundle["model"]
            cluster_to_segment = bundle["cluster_to_segment"]
            mode = bundle["mode"]

            sub_single, sub_batch = st.tabs(["Single record", "Batch CSV upload"])

            with sub_single:
                if mode == "rfm":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
                    with col2:
                        frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
                    with col3:
                        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

                    if st.button("Predict Segment", key="lookup_rfm"):
                        cluster, segment, _ = utils.predict_single_customer(
                            recency, frequency, monetary, preprocessor, pca, kmeans, cluster_to_segment
                        )
                        st.markdown(f"### Predicted segment: <span class='segment-badge'>{segment}</span>",
                                    unsafe_allow_html=True)
                        st.write(utils.SEGMENT_DESCRIPTIONS.get(segment, ""))
                else:
                    numeric_cols = bundle["numeric_cols"]
                    categorical_cols = bundle["categorical_cols"]
                    feature_values = {}
                    cols = st.columns(2)
                    with cols[0]:
                        for nc in numeric_cols:
                            feature_values[nc] = st.number_input(nc, value=0.0, key=f"num_{nc}")
                    with cols[1]:
                        for cc in categorical_cols:
                            feature_values[cc] = st.text_input(cc, key=f"cat_{cc}")

                    if st.button("Predict Segment", key="lookup_generic"):
                        cluster, segment, _ = utils.predict_single_generic(
                            feature_values, numeric_cols, categorical_cols, preprocessor, pca, kmeans, cluster_to_segment
                        )
                        st.markdown(f"### Predicted segment: <span class='segment-badge'>{segment}</span>",
                                    unsafe_allow_html=True)

            with sub_batch:
                uploaded_batch = st.file_uploader("Upload a CSV to score", type=["csv"], key="lookup_batch")
                if uploaded_batch is not None:
                    try:
                        batch_df = pd.read_csv(uploaded_batch, encoding="ISO-8859-1")
                        if mode == "rfm":
                            result = utils.predict_segments_for_raw_data(
                                batch_df, preprocessor, pca, kmeans, cluster_to_segment
                            )
                        else:
                            result = utils.predict_segments_for_raw_data_generic(
                                batch_df, bundle["numeric_cols"], bundle["categorical_cols"],
                                preprocessor, pca, kmeans, cluster_to_segment
                            )
                        st.success(f"Scored {len(result):,} records.")
                        st.dataframe(result, use_container_width=True)
                        st.download_button(
                            "Download scored CSV", data=result.to_csv(index=False).encode("utf-8"),
                            file_name="scored_batch.csv", mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Couldn't process this file: {e}")

# ==========================================================================
# TAB 3 — About
# ==========================================================================

with tab_about:
    st.subheader("Algorithms")
    st.markdown("""
- **K-Means** — splits customers into exactly *k* groups by minimizing distance to each
  group's center. Fast, easy to interpret, and the only one here that can score brand-new
  customers later without refitting.
- **Hierarchical (Agglomerative)** — merges customers into a tree of nested clusters, then
  cuts the tree into *k* groups. Doesn't require picking *k* up front (you can inspect the
  tree), but doesn't scale well to large datasets and can't score new data afterward.
- **DBSCAN** — groups points that are densely packed together and marks sparse points as
  *noise/outliers* (labeled -1). Doesn't require picking the number of clusters at all — it's
  driven by `eps` and `min_samples` — but also can't score new data afterward.
    """)

    st.subheader("Why can't Hierarchical/DBSCAN score new data?")
    st.write(
        "Both algorithms only produce labels for the exact points they were fit on — "
        "scikit-learn doesn't implement a `.predict()` for unseen points for either one. "
        "K-Means, by contrast, keeps cluster centers around, so `.predict()` just finds the "
        "nearest center for a new point. That's why only K-Means models can be saved and "
        "reused in the Quick Lookup tab."
    )

    st.subheader("Segment naming")
    st.write(
        "For retail/RFM data, segments are ranked and named using standard RFM tiers "
        "(Champions, Loyal Customers, Potential Loyalists, At Risk, Hibernating). "
        "For manually-selected columns, each cluster is named from whichever 1-2 numeric "
        "columns deviate most from the overall average (e.g. 'High Income, Low Age')."
    )
    for name, desc in utils.SEGMENT_DESCRIPTIONS.items():
        st.markdown(f"**{name}** — {desc}")
