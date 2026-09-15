# Customer Segmentation — Interactive Multi-Algorithm Dashboard

An end-to-end customer segmentation project built around **RFM analysis**
(Recency, Frequency, Monetary) with a choice of **three clustering
algorithms** — K-Means, Hierarchical (Agglomerative), and DBSCAN — plus
**PCA** for visualization, and an interactive **Streamlit** dashboard where
you upload a CSV and walk through every step yourself.

## Project structure

```
Customer-Segmentation/
│
├── data/
│     └── online_retail.csv         # raw transactional data (already included)
│
├── notebooks/
│     └── Customer_Segmentation.ipynb   # offline training pipeline + EDA (algorithm-selectable)
│
├── models/
│     ├── cluster_model.pkl   # {"algorithm", "mode", "model", "cluster_to_segment", ...}
│     ├── scaler.pkl          # fitted StandardScaler (RFM mode) or ColumnTransformer (generic mode)
│     └── pca.pkl             # fitted PCA(n_components=2)
│
├── app.py                # Streamlit dashboard (the main way to use this project)
├── utils.py               # shared feature engineering / clustering / model helpers
├── requirements.txt
└── README.md
```

## How the app works

`app.py` has three tabs:

### 1. 🧪 Upload & Cluster (Live)

The main workflow. Upload **any** CSV:

- If it looks like transactional retail data (`InvoiceNo`, `InvoiceDate`,
  `Quantity`, `UnitPrice`, `CustomerID`, ...), RFM features are built for you
  automatically. You can override this and pick columns manually if you want.
- Otherwise, you manually pick which columns are numeric and which are
  categorical — those become the clustering features (imputed, scaled/
  one-hot-encoded automatically).

Then, step by step in the UI:
1. Choose an algorithm: **K-Means**, **Hierarchical (Agglomerative)**, or **DBSCAN**
2. Set its parameters (with built-in diagnostics — elbow/silhouette for
   K-Means/Hierarchical, a k-distance plot for DBSCAN's `eps`)
3. Click **Run Clustering** to see cluster counts, a PCA scatter plot colored
   by segment, a scored table, and a CSV download
4. Optionally **save the pipeline** so it can be reused in the Quick Lookup tab

### 2. 🔎 Quick Lookup (Saved Model)

Reuses whatever model is currently saved in `models/` — either from running
the notebook, or from clicking "save" in the Upload & Cluster tab — to score
a single new record or a new batch CSV.

### 3. ℹ️ About

Explains the three algorithms, why only K-Means supports this kind of reuse,
and how segments get their names.

## ⚠️ Important limitation: only K-Means can score new data later

This is a scikit-learn limitation, not a design choice:

- **K-Means** keeps its cluster centers around after fitting, so `.predict()`
  on a brand-new point just finds the nearest center. This is what makes
  saved-model reuse possible.
- **Hierarchical (Agglomerative)** and **DBSCAN** only ever produce labels
  for the exact data they were fit on — scikit-learn doesn't implement
  `.predict()` for unseen points for either one. Every time you want cluster
  labels with these two, you re-run them on the dataset you want labeled.

So: **Upload & Cluster (Live)** lets you use and compare all three algorithms
freely, but the **Quick Lookup** tab (single-record and new-batch scoring)
only works when the currently saved model is K-Means. If you save a
Hierarchical or DBSCAN result, the app will tell you clearly rather than
silently doing something wrong.

## Dataset

`data/online_retail.csv` is the **UCI "Online Retail" dataset** — a
transactional log of a UK-based online retailer, Dec 2010–Dec 2011, ~541,000
transactions across ~4,300 customers. Already included, nothing to download.

| Column       | Description                                   |
|--------------|------------------------------------------------|
| InvoiceNo    | Invoice number (prefixed with `C` = cancelled) |
| StockCode    | Product code                                   |
| Description  | Product name                                   |
| Quantity     | Units purchased                                |
| InvoiceDate  | Date/time of transaction                       |
| UnitPrice    | Price per unit                                 |
| CustomerID   | Unique customer identifier                     |
| Country      | Customer's country                             |

You can also drag in any other CSV (customer demographics, survey data,
whatever) — the app will fall back to manual column selection automatically.

## Setup

```bash
git clone <your-repo-url>
cd Customer-Segmentation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Option A — Just use the app (recommended)

```bash
streamlit run app.py
```

Go to **Upload & Cluster (Live)**, upload `data/online_retail.csv` (or your
own data), pick an algorithm, and run it. No notebook or pre-trained files
needed for this tab.

## Option B — Train offline in the notebook first

```bash
jupyter notebook notebooks/Customer_Segmentation.ipynb
```

**Important:** launch Jupyter from inside this same activated `venv` (the one
where you ran `pip install -r requirements.txt`), or you risk training with a
different scikit-learn version than the one `app.py` loads models with later
— which will crash the app with an `InconsistentVersionWarning` /
`AttributeError`. Check with:
```python
import sklearn
print(sklearn.__version__)
```
and compare to `pip show scikit-learn` in your terminal.

Set `ALGORITHM = "kmeans" | "hierarchical" | "dbscan"` near the top of the
clustering section and run all cells. This saves `scaler.pkl`, `pca.pkl`, and
`cluster_model.pkl` into `models/` for the **Quick Lookup** tab (only useful
if `ALGORITHM == "kmeans"` — see the limitation above).

## Segments

For RFM/retail data, segments are ranked and named from each cluster's
average Recency/Frequency/Monetary:

| Segment | Description |
|---|---|
| **Champions** | Recent, frequent, high spend. Reward, upsell, ask for reviews. |
| **Loyal Customers** | Buy regularly, respond well to engagement. Upsell, ask for referrals. |
| **Potential Loyalists** | Recent with decent frequency/spend. Nudge into a loyalty program. |
| **At Risk / Need Attention** | Below-average across the board. Reactivate with targeted offers. |
| **Hibernating / Lost** | Long time since last purchase. Win-back campaign or deprioritize. |
| **Noise / Outliers** | DBSCAN-only: points that didn't fit densely into any cluster. |

For manually-selected columns (generic mode), each cluster is named from
whichever 1–2 numeric columns deviate most from the overall average (e.g.
*"High Income, Low Age"*).

## Notes

- `utils.py` is imported by **both** the notebook and `app.py`, so the exact
  same feature-building/clustering logic is used everywhere.
- Cluster **numbers** are arbitrary between reruns; segment **names** (not
  raw cluster IDs) are what's saved and used for lookups.
- Hierarchical clustering is O(n²) in memory/time — the app automatically
  offers to sample down to 5,000 rows if your dataset is larger.
- `requirements.txt` pins `scikit-learn==1.8.0` exactly so the version used
  to train models always matches the version used to load them.
