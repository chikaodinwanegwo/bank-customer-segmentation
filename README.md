# ML Customer Segmentation for Banks: From "Same Email for Everyone" to Surgical Retention

Banks spend heavily to acquire customers, then treat a $50K saver and a $5M portfolio holder identically.
This project shows how unsupervised ML can split a client base into **actionable segments** and attach a
**next-best-action playbook** to each one.

> Uses **synthetic data** (5,000 customers) so it can be shared publicly. The pipeline works the same on real CRM/core-banking extracts.

## Segments discovered
| Segment | Signature | Action |
|---|---|---|
| High-net-worth loyalist | Multi-product, long tenure, 3+ referrals | White-glove relationship management |
| Wealth builder | Deposits growing fast, few products | Proactive advisor call, investment offer |
| Silent churner | Logins and balance falling, ~4 months idle | Personalised re-engagement now |
| Rate shopper | Moves money when rates change | Loyalty rate incentive, not a newsletter |
| Early-career client | Young, low balance, short tenure | Nurture for lifetime value |

## Key findings (from the run)
- ~10% of customers (HNW loyalists) hold **~71% of balances**, yet get the same newsletter as everyone else.
- Early-career clients are **30% of customers but <1% of balances**: ignore them on AUM, and you lose 30 years of value.
- Silent churners hold **~$20M** in balances and show the clearest warning signs (-58% logins, -25% balances).

## How it works
1. `generate_data.py`: creates synthetic customers from 5 hidden behavioural archetypes
2. `segment.py`:
   - feature engineering (log-balance, trend features), scaling
   - K-Means with silhouette analysis to evaluate k
   - business labelling of clusters by matching centroid profiles to segment definitions
   - maps each segment to an action playbook
   - outputs CSVs and charts

## Run it
```bash
pip install -r requirements.txt
python generate_data.py
python segment.py
```
Outputs land in `outputs/`: `customers_segmented.csv`, `segment_summary.csv`, `segments.png`, `segment_profiles.png`.

## Honest limitations / next steps
- k=5 was chosen for **business interpretability**; silhouette alone slightly favours k=2-3. Always validate with stakeholders.
- Synthetic data is cleaner than reality; real data needs missing-value handling and outlier treatment.
- Next: add a churn-prediction model (XGBoost) per segment, CLV estimates, and a Streamlit dashboard.
