# Customer Segmentation — Online Retail Dataset

An end-to-end customer segmentation project using **RFM analysis**
(Recency, Frequency, Monetary), **KMeans clustering**, and **PCA**,
with a **Streamlit** dashboard for interactive scoring.

## Project structure

```
Customer-Segmentation/
│
├── data/
│     └── online_retail.csv         # raw transactional data (already included)
│
├── notebooks/
│     └── Customer_Segmentation.ipynb   # full training pipeline + EDA
│
├── models/
│     ├── kmeans.pkl                # {"model": KMeans, "cluster_to_segment": {...}}
│     ├── scaler.pkl                # fitted StandardScaler
│     └── pca.pkl                   # fitted PCA(n_components=2)
│
├── app.py                          # Streamlit dashboard
├── utils.py                        # shared feature engineering / model helpers
├── requirements.txt
└── README.md
```

## Dataset

This is the **UCI "Online Retail" dataset** — a transactional log of a
UK-based online retailer, Dec 2010–Dec 2011, ~541,000 transactions across
~4,300 customers. It's already included at `data/online_retail.csv`.

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

## Setup

```bash
git clone <your-repo-url>
cd Customer-Segmentation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`data/online_retail.csv` is already in the project — no need to download anything.

## 1. Train the models (run the notebook)

```bash
jupyter notebook notebooks/Customer_Segmentation.ipynb
```

**Important:** make sure Jupyter is launched from inside this same activated
`venv` (the one where you just ran `pip install -r requirements.txt`). If
Jupyter opens with a different Python kernel, the models will be pickled
with a different scikit-learn version than the one `app.py` loads them
with later, which will crash the app with an `InconsistentVersionWarning` /
`AttributeError`. You can check inside the notebook with:
```python
import sklearn
print(sklearn.__version__)
```
and compare it to `pip show scikit-learn` in your terminal — they should match.

Run all cells top to bottom. This will:
1. Load and clean the raw data
2. Engineer RFM features per customer
3. Explore distributions (EDA)
4. Log-transform + scale features
5. Pick a cluster count via the elbow method / silhouette score
6. Fit the final KMeans model
7. Profile clusters and assign segment names (Champions, Loyal Customers, etc.)
8. Visualize clusters via PCA
9. Save `scaler.pkl`, `pca.pkl`, and `kmeans.pkl` into `models/`

## 2. Run the dashboard

```bash
streamlit run app.py
```

The app has three tabs:

- **Single Customer Lookup** — manually enter Recency/Frequency/Monetary and
  see the predicted segment.
- **Batch Scoring (CSV Upload)** — upload a raw transactional CSV (same schema
  as `online_retail.csv` — you can re-upload `data/online_retail.csv` itself
  to try it) and get every customer segmented, with charts and a
  downloadable results CSV.
- **About the Segments** — explanation of each segment and the current
  cluster → segment mapping.

## Segments

Segments are named heuristically by ranking each cluster's average R/F/M
against the other clusters:

| Segment | Description |
|---|---|
| **Champions** | Recent, frequent, high spend. Reward, upsell, ask for reviews. |
| **Loyal Customers** | Buy regularly, respond well to engagement. Upsell, ask for referrals. |
| **Potential Loyalists** | Recent with decent frequency/spend. Nudge into a loyalty program. |
| **At Risk / Need Attention** | Below-average across the board. Reactivate with targeted offers. |
| **Hibernating / Lost** | Long time since last purchase. Win-back campaign or deprioritize. |

The exact mapping is recomputed at training time based on your data's actual
cluster centers (see `utils.label_segments`).

## Notes

- `utils.py` is imported by **both** the notebook and `app.py`, so the exact
  same cleaning/RFM/scaling/PCA logic is used at training and inference time.
- KMeans cluster **numbers** are arbitrary between reruns; that's why the
  segment **names** (not raw cluster IDs) are what's saved and used in the app.
- To change the number of clusters, edit `N_CLUSTERS` in the notebook and
  re-run.
- `requirements.txt` pins `scikit-learn==1.8.0` exactly (rather than `>=`) so
  the version used to train the models always matches the version used to
  load them in the app.
