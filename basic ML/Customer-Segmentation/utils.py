"""
utils.py
--------
Shared helper functions for the Customer Segmentation project.

Two feature-building paths:

  1. RETAIL / RFM PATH — auto-detected when the uploaded CSV looks like the
     Online Retail transactional schema (InvoiceNo, InvoiceDate, Quantity,
     UnitPrice, CustomerID, ...). Builds Recency/Frequency/Monetary features
     per customer, same as before.

  2. GENERIC PATH — used for any other CSV. The user manually picks which
     columns are numeric and which are categorical; those columns are
     imputed, scaled/one-hot-encoded, and used as clustering features.

Three clustering algorithms are supported: KMeans, Agglomerative
(Hierarchical), and DBSCAN. Important asymmetry to be aware of:

  - KMeans has a real `.predict()` method, so a saved KMeans model can score
    brand-new individual customers or new batches later without refitting.
  - Agglomerative and DBSCAN do NOT support predicting new/unseen points in
    scikit-learn — they only produce labels for the exact data they were fit
    on. So those two are "batch-only": every time you want cluster labels
    for a dataset, you refit on that dataset. Saving one of these models is
    still useful for re-inspecting results later, but not for scoring new
    single customers or new batches (`supports_predict` reflects this).

These functions are used both by the training notebook
(notebooks/Customer_Segmentation.ipynb) and the Streamlit app (app.py).
"""

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Retail schema detection
# --------------------------------------------------------------------------

REQUIRED_RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

# Columns that MUST be present (case-insensitively) for RFM auto-detection.
# "Description" and "Country" are nice-to-have but not required.
RFM_CORE_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]


def detect_retail_schema(df: pd.DataFrame) -> bool:
    """
    Return True if the dataframe looks like transactional Online-Retail-style
    data (has the core columns needed to compute RFM), regardless of column
    name casing/whitespace.
    """
    normalized = {c.strip().lower() for c in df.columns}
    required = {c.lower() for c in RFM_CORE_COLUMNS}
    return required.issubset(normalized)


# --------------------------------------------------------------------------
# RETAIL / RFM PATH
# --------------------------------------------------------------------------

def load_data(filepath: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load a raw CSV. Works for both the retail transactional file and generic CSVs."""
    return pd.read_csv(filepath, encoding=encoding)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail transactional data.

    Steps:
      - Drop rows with missing CustomerID
      - Remove cancelled orders (InvoiceNo starting with 'C')
      - Remove non-positive Quantity / UnitPrice
      - Drop exact duplicate rows
      - Create a TotalPrice column
    """
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]

    data = data.dropna(subset=["CustomerID"])
    data["CustomerID"] = data["CustomerID"].astype(int).astype(str)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data = data.drop_duplicates()
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    return data


def calculate_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Build an RFM (Recency, Frequency, Monetary) table, one row per CustomerID."""
    data = df.copy()

    if snapshot_date is None:
        snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def log_transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Log1p Recency/Frequency/Monetary to reduce right-skew before scaling."""
    transformed = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        transformed[col] = np.log1p(transformed[col].clip(lower=0))
    return transformed


def scale_features(feature_df: pd.DataFrame, scaler=None, fit: bool = True, cols=None):
    """
    Scale numeric columns with a StandardScaler.
    Pass an existing fitted `scaler` and fit=False at prediction time.
    """
    from sklearn.preprocessing import StandardScaler

    if cols is None:
        cols = ["Recency", "Frequency", "Monetary"]

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaled_values = scaler.fit_transform(feature_df[cols])
    else:
        scaled_values = scaler.transform(feature_df[cols])

    scaled_df = pd.DataFrame(scaled_values, columns=cols, index=feature_df.index)
    return scaled_df, scaler


def build_rfm_features(raw_df: pd.DataFrame):
    """
    Full RFM feature-building shortcut used by the auto-detected retail path:
    clean -> RFM -> log-transform -> (unscaled) feature table indexed by CustomerID.
    """
    cleaned = clean_data(raw_df)
    rfm = calculate_rfm(cleaned)
    rfm_indexed = rfm.set_index("CustomerID")
    log_rfm = log_transform_rfm(rfm_indexed)
    return log_rfm, rfm_indexed  # (features to scale, original human-readable RFM)


# --------------------------------------------------------------------------
# GENERIC PATH (manual column selection)
# --------------------------------------------------------------------------

def clean_generic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning for arbitrary uploaded CSVs: drop exact duplicates only."""
    return df.drop_duplicates().copy()


def build_preprocessor(numeric_cols, categorical_cols):
    """
    A ColumnTransformer that:
      - median-imputes + standard-scales the given numeric columns
      - most-frequent-imputes + one-hot-encodes the given categorical columns

    Parameterized by column lists chosen by the user in the UI (unlike a
    fixed schema), so it works for any uploaded CSV.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def preprocess_features(df: pd.DataFrame, numeric_cols, categorical_cols, preprocessor=None, fit: bool = True):
    """
    Run the ColumnTransformer over the chosen feature columns.
    Pass an existing fitted `preprocessor` and fit=False at prediction time.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df[numeric_cols + categorical_cols]

    if fit:
        transformed = preprocessor.fit_transform(X)
    else:
        transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    return transformed_df, preprocessor


# --------------------------------------------------------------------------
# PCA (shared by both paths)
# --------------------------------------------------------------------------

def apply_pca(transformed_df: pd.DataFrame, pca=None, fit: bool = True, n_components: int = 2):
    """Reduce transformed features to n_components with PCA (for visualization)."""
    from sklearn.decomposition import PCA

    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)

    if fit:
        components = pca.fit_transform(transformed_df)
    else:
        components = pca.transform(transformed_df)

    cols = [f"PC{i+1}" for i in range(components.shape[1])]
    pca_df = pd.DataFrame(components, columns=cols, index=transformed_df.index)
    return pca_df, pca


# --------------------------------------------------------------------------
# Clustering algorithms
# --------------------------------------------------------------------------

ALGORITHMS = ["kmeans", "hierarchical", "dbscan"]

ALGORITHM_LABELS = {
    "kmeans": "K-Means",
    "hierarchical": "Hierarchical (Agglomerative)",
    "dbscan": "DBSCAN",
}

# Whether a fitted model of this algorithm can score NEW/unseen data later.
SUPPORTS_PREDICT = {
    "kmeans": True,
    "hierarchical": False,
    "dbscan": False,
}


def train_kmeans(feature_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(feature_df)
    return model, labels


def train_hierarchical(feature_df: pd.DataFrame, n_clusters: int = 4, linkage: str = "ward"):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(feature_df)
    return model, labels


def train_dbscan(feature_df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(feature_df)  # -1 = noise / outlier
    return model, labels


def run_clustering(algorithm: str, feature_df: pd.DataFrame, **params):
    """
    Dispatch to the chosen algorithm. `params` should match the algorithm:
      kmeans:       n_clusters
      hierarchical: n_clusters, linkage
      dbscan:       eps, min_samples

    Returns (model, labels, supports_predict).
    """
    if algorithm == "kmeans":
        model, labels = train_kmeans(feature_df, n_clusters=params.get("n_clusters", 4))
    elif algorithm == "hierarchical":
        model, labels = train_hierarchical(
            feature_df,
            n_clusters=params.get("n_clusters", 4),
            linkage=params.get("linkage", "ward"),
        )
    elif algorithm == "dbscan":
        model, labels = train_dbscan(
            feature_df,
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, labels, SUPPORTS_PREDICT[algorithm]


def compute_k_distance(feature_df: pd.DataFrame, k: int = 5):
    """
    Sorted distance-to-k-th-nearest-neighbor for every point — used to plot
    a "k-distance graph" that helps pick a reasonable DBSCAN `eps` (the
    elbow of this sorted curve is a good starting eps).
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(feature_df)
    distances, _ = nbrs.kneighbors(feature_df)
    k_distances = np.sort(distances[:, -1])
    return k_distances


def elbow_and_silhouette(feature_df: pd.DataFrame, k_range=range(2, 10)):
    """
    Inertia (KMeans only) + silhouette score across a range of k, to help
    pick the number of clusters for KMeans / Hierarchical.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feature_df)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(feature_df, labels))

    return list(k_range), inertias, silhouettes


# --------------------------------------------------------------------------
# Segment naming — RFM path (unchanged behavior)
# --------------------------------------------------------------------------

def label_segments(rfm_with_clusters: pd.DataFrame, cluster_col: str = "Cluster") -> dict:
    """
    Given an RFM table with a cluster assignment column, rank clusters by
    mean Recency/Frequency/Monetary and heuristically name them. DBSCAN's
    noise cluster (-1), if present, is always named 'Noise / Outliers' and
    excluded from the ranking of the "real" clusters.
    """
    working = rfm_with_clusters.copy()
    is_noise_present = -1 in working[cluster_col].unique()
    ranked_df = working[working[cluster_col] != -1] if is_noise_present else working

    mapping = {}
    if len(ranked_df) > 0:
        profile = ranked_df.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

        r_rank = profile["Recency"].rank(ascending=True)
        f_rank = profile["Frequency"].rank(ascending=False)
        m_rank = profile["Monetary"].rank(ascending=False)
        score = r_rank + f_rank + m_rank

        ordered_clusters = score.sort_values().index.tolist()

        names_best_to_worst = [
            "Champions", "Loyal Customers", "Potential Loyalists",
            "At Risk / Need Attention", "Hibernating / Lost",
        ]
        n = len(ordered_clusters)
        chosen_names = (
            names_best_to_worst[:n] if n <= len(names_best_to_worst)
            else names_best_to_worst + [f"Segment {i}" for i in range(n - len(names_best_to_worst))]
        )
        mapping = {cid: name for cid, name in zip(ordered_clusters, chosen_names)}

    if is_noise_present:
        mapping[-1] = "Noise / Outliers"

    return mapping


SEGMENT_DESCRIPTIONS = {
    "Champions": "Bought recently, buy often, and spend the most. Reward them, ask for reviews, upsell new products.",
    "Loyal Customers": "Buy regularly and respond well to engagement. Upsell higher-value products, ask for referrals.",
    "Potential Loyalists": "Recent customers with decent frequency/spend. Offer membership or loyalty programs to build habit.",
    "At Risk / Need Attention": "Below-average recency, frequency, and monetary. Reactivate with personalized offers and reminders.",
    "Hibernating / Lost": "Long time since last purchase, low frequency and spend. Win-back campaigns or let go to focus budget elsewhere.",
    "Noise / Outliers": "DBSCAN could not confidently assign these points to a dense cluster — they sit apart from the main groups.",
}


# --------------------------------------------------------------------------
# Segment naming — GENERIC path (any user-chosen columns)
# --------------------------------------------------------------------------

def _mode_or_unknown(series: pd.Series) -> str:
    m = series.mode()
    return m.iat[0] if not m.empty else "Unknown"


def label_segments_generic(df_with_clusters: pd.DataFrame, numeric_cols, categorical_cols, cluster_col: str = "Cluster"):
    """
    Generic cluster naming for arbitrary feature columns: for each cluster,
    find the 1-2 numeric columns whose cluster mean deviates most (in
    z-score terms) from the overall mean, and describe the cluster as
    "High <col>, Low <col>" etc. Falls back to the most common categorical
    value if there are no numeric columns. DBSCAN noise (-1) is always
    labeled 'Noise / Outliers'.

    Returns (mapping, profile_df).
    """
    working = df_with_clusters.copy()
    is_noise_present = -1 in working[cluster_col].unique()
    real_df = working[working[cluster_col] != -1] if is_noise_present else working

    mapping = {}
    profile = pd.DataFrame(index=real_df[cluster_col].unique())

    if len(real_df) > 0 and numeric_cols:
        cluster_means = real_df.groupby(cluster_col)[numeric_cols].mean()
        overall_mean = real_df[numeric_cols].mean()
        overall_std = real_df[numeric_cols].std().replace(0, 1)

        z_scores = (cluster_means - overall_mean) / overall_std
        profile = cluster_means.copy()

        for cid in cluster_means.index:
            row_z = z_scores.loc[cid].sort_values(key=lambda s: s.abs(), ascending=False)
            top_feats = row_z.head(2)
            parts = []
            for col, z in top_feats.items():
                direction = "High" if z > 0 else "Low"
                parts.append(f"{direction} {col}")
            name = ", ".join(parts) if parts else f"Cluster {cid}"
            mapping[cid] = name
    elif len(real_df) > 0 and categorical_cols:
        modes = real_df.groupby(cluster_col)[categorical_cols[0]].agg(_mode_or_unknown)
        profile = modes.to_frame()
        for cid, val in modes.items():
            mapping[cid] = f"Mostly {val}"
    else:
        for cid in real_df[cluster_col].unique():
            mapping[cid] = f"Cluster {cid}"

    if is_noise_present:
        mapping[-1] = "Noise / Outliers"

    return mapping, profile


# --------------------------------------------------------------------------
# Batch scoring / single prediction against a SAVED KMeans model
# (only KMeans supports out-of-sample predict — see module docstring)
# --------------------------------------------------------------------------

def predict_segments_for_raw_data(raw_df: pd.DataFrame, scaler, pca, kmeans, cluster_to_segment: dict):
    """RFM-path batch scoring against a saved, fitted KMeans model."""
    cleaned = clean_data(raw_df)
    rfm = calculate_rfm(cleaned)
    log_rfm = log_transform_rfm(rfm.set_index("CustomerID"))
    scaled_df, _ = scale_features(log_rfm, scaler=scaler, fit=False)
    pca_df, _ = apply_pca(scaled_df, pca=pca, fit=False)

    clusters = kmeans.predict(scaled_df)

    result = rfm.copy()
    result["Cluster"] = clusters
    result["Segment"] = result["Cluster"].map(cluster_to_segment)
    result["PC1"] = pca_df["PC1"].values
    result["PC2"] = pca_df["PC2"].values
    return result


def predict_single_customer(recency: float, frequency: float, monetary: float,
                             scaler, pca, kmeans, cluster_to_segment: dict):
    """RFM-path single-customer scoring against a saved, fitted KMeans model."""
    single = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
    log_single = log_transform_rfm(single)
    scaled_df, _ = scale_features(log_single, scaler=scaler, fit=False)
    pca_df, _ = apply_pca(scaled_df, pca=pca, fit=False)
    cluster = int(kmeans.predict(scaled_df)[0])
    segment = cluster_to_segment.get(cluster, f"Segment {cluster}")
    return cluster, segment, pca_df.iloc[0].to_dict()


def predict_segments_for_raw_data_generic(raw_df: pd.DataFrame, numeric_cols, categorical_cols,
                                           preprocessor, pca, kmeans, cluster_to_segment: dict):
    """Generic-path batch scoring against a saved, fitted KMeans model."""
    cleaned = clean_generic_data(raw_df)
    transformed_df, _ = preprocess_features(cleaned, numeric_cols, categorical_cols, preprocessor=preprocessor, fit=False)
    pca_df, _ = apply_pca(transformed_df, pca=pca, fit=False)

    clusters = kmeans.predict(transformed_df)

    result = cleaned.copy()
    result["Cluster"] = clusters
    result["Segment"] = result["Cluster"].map(cluster_to_segment)
    result["PC1"] = pca_df["PC1"].values
    result["PC2"] = pca_df["PC2"].values
    return result


def predict_single_generic(feature_values: dict, numeric_cols, categorical_cols,
                            preprocessor, pca, kmeans, cluster_to_segment: dict):
    """Generic-path single-record scoring against a saved, fitted KMeans model."""
    single = pd.DataFrame([feature_values])
    transformed_df, _ = preprocess_features(single, numeric_cols, categorical_cols, preprocessor=preprocessor, fit=False)
    pca_df, _ = apply_pca(transformed_df, pca=pca, fit=False)
    cluster = int(kmeans.predict(transformed_df)[0])
    segment = cluster_to_segment.get(cluster, f"Segment {cluster}")
    return cluster, segment, pca_df.iloc[0].to_dict()


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------

def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
