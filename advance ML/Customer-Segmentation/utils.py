"""
utils.py
--------
Shared helper functions for the Customer Segmentation project.

This version works on the real UCI Online Retail transactional dataset
(data/online_retail.csv) — columns:

    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
    CustomerID, Country

These functions build an RFM (Recency, Frequency, Monetary) profile per
customer, then scale + PCA-reduce + cluster them with KMeans. The same
functions are used both by the training notebook
(notebooks/Customer_Segmentation.ipynb) and the Streamlit app (app.py), so
the exact same preprocessing is used at training time and prediction time.
"""

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------

REQUIRED_RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def load_data(filepath: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load the raw Online Retail transactional CSV."""
    df = pd.read_csv(filepath, encoding=encoding)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Online Retail transactional data.

    Steps:
      - Drop rows with missing CustomerID (can't attribute to a customer)
      - Remove cancelled orders (InvoiceNo starting with 'C')
      - Remove non-positive Quantity / UnitPrice (returns, adjustments, errors)
      - Drop exact duplicate rows
      - Create a TotalPrice column
    """
    data = df.copy()

    # Standardize column names in case of stray whitespace
    data.columns = [c.strip() for c in data.columns]

    # Drop missing CustomerID
    data = data.dropna(subset=["CustomerID"])

    # Ensure correct dtypes
    data["CustomerID"] = data["CustomerID"].astype(int).astype(str)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    # Remove cancellations (InvoiceNo starting with 'C')
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove non-positive quantities/prices
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

    # Drop duplicates
    data = data.drop_duplicates()

    # Total price per line item
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    return data


# --------------------------------------------------------------------------
# RFM feature engineering
# --------------------------------------------------------------------------

def calculate_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Build an RFM (Recency, Frequency, Monetary) table, one row per CustomerID.

    Recency  = days since the customer's most recent purchase (relative to snapshot_date)
    Frequency = number of distinct invoices (orders) placed
    Monetary  = total amount spent
    """
    data = df.copy()

    if snapshot_date is None:
        snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


# --------------------------------------------------------------------------
# Preprocessing: log transform + scaling + PCA
# --------------------------------------------------------------------------

def log_transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    RFM features are heavily right-skewed. Apply log1p to Recency, Frequency,
    and Monetary to make them closer to normal before scaling/clustering.
    Kept separate from calculate_rfm so the raw RFM table stays human-readable.
    """
    transformed = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        transformed[col] = np.log1p(transformed[col].clip(lower=0))
    return transformed


def scale_features(feature_df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    Scale Recency/Frequency/Monetary columns with a StandardScaler.
    Pass an existing fitted `scaler` and fit=False at prediction time.
    """
    from sklearn.preprocessing import StandardScaler

    cols = ["Recency", "Frequency", "Monetary"]
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaled_values = scaler.fit_transform(feature_df[cols])
    else:
        scaled_values = scaler.transform(feature_df[cols])

    scaled_df = pd.DataFrame(scaled_values, columns=cols, index=feature_df.index)
    return scaled_df, scaler


def apply_pca(scaled_df: pd.DataFrame, pca=None, fit: bool = True, n_components: int = 2):
    """
    Reduce scaled RFM features to n_components with PCA (used for 2D
    visualization of clusters, and optionally as clustering input).
    """
    from sklearn.decomposition import PCA

    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)

    if fit:
        components = pca.fit_transform(scaled_df)
    else:
        components = pca.transform(scaled_df)

    cols = [f"PC{i+1}" for i in range(components.shape[1])]
    pca_df = pd.DataFrame(components, columns=cols, index=scaled_df.index)
    return pca_df, pca


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def train_kmeans(scaled_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    """Fit a KMeans model on scaled RFM features."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_df)
    return kmeans, labels


# --------------------------------------------------------------------------
# Segment naming
# --------------------------------------------------------------------------

def label_segments(rfm_with_clusters: pd.DataFrame, cluster_col: str = "Cluster") -> dict:
    """
    Given an RFM table that includes a cluster assignment column, compute
    the mean R/F/M per cluster and heuristically assign a human-readable
    segment name to each cluster ID.

    Returns a dict: {cluster_id: segment_name}
    """
    profile = rfm_with_clusters.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

    # Rank clusters on each dimension (lower Recency is better, higher F/M is better)
    r_rank = profile["Recency"].rank(ascending=True)    # 1 = most recent (best)
    f_rank = profile["Frequency"].rank(ascending=False)  # 1 = most frequent (best)
    m_rank = profile["Monetary"].rank(ascending=False)   # 1 = highest spend (best)

    score = r_rank + f_rank + m_rank  # lower total = better customer

    ordered_clusters = score.sort_values().index.tolist()

    names_best_to_worst = [
        "Champions",         # recent, frequent, high spend
        "Loyal Customers",   # frequent, decent spend
        "Potential Loyalists",  # fairly recent, moderate F/M
        "At Risk / Need Attention",
        "Hibernating / Lost",
    ]

    # Map based on rank position, reusing/truncating names if cluster count differs
    n = len(ordered_clusters)
    if n <= len(names_best_to_worst):
        chosen_names = names_best_to_worst[:n]
    else:
        chosen_names = names_best_to_worst + [f"Segment {i}" for i in range(n - len(names_best_to_worst))]

    mapping = {cluster_id: name for cluster_id, name in zip(ordered_clusters, chosen_names)}
    return mapping


SEGMENT_DESCRIPTIONS = {
    "Champions": "Bought recently, buy often, and spend the most. Reward them, ask for reviews, upsell new products.",
    "Loyal Customers": "Buy regularly and respond well to engagement. Upsell higher-value products, ask for referrals.",
    "Potential Loyalists": "Recent customers with decent frequency/spend. Offer membership or loyalty programs to build habit.",
    "At Risk / Need Attention": "Below-average recency, frequency, and monetary. Reactivate with personalized offers and reminders.",
    "Hibernating / Lost": "Long time since last purchase, low frequency and spend. Win-back campaigns or let go to focus budget elsewhere.",
}


# --------------------------------------------------------------------------
# Full pipeline for a batch of new/raw transactional data (used by the app)
# --------------------------------------------------------------------------

def predict_segments_for_raw_data(raw_df: pd.DataFrame, scaler, pca, kmeans, cluster_to_segment: dict):
    """
    Take raw transactional data (same schema as online_retail.csv), run it
    through the full feature pipeline using ALREADY-FITTED scaler/pca/kmeans,
    and return a per-customer RFM + cluster + segment table.
    """
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
    """Predict the segment for one manually-entered customer (R/F/M values)."""
    single = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
    log_single = log_transform_rfm(single)
    scaled_df, _ = scale_features(log_single, scaler=scaler, fit=False)
    pca_df, _ = apply_pca(scaled_df, pca=pca, fit=False)
    cluster = int(kmeans.predict(scaled_df)[0])
    segment = cluster_to_segment.get(cluster, f"Segment {cluster}")
    return cluster, segment, pca_df.iloc[0].to_dict()


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------

def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
