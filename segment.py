"""ML customer segmentation + next-best-action playbook."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/customers.csv")

# 1. Features (log-transform balance because it is heavily skewed)
df["log_balance"] = np.log(df["total_balance"])
FEATURES = ["age", "tenure_years", "log_balance", "num_products", "balance_change_6m_pct",
            "logins_per_month", "login_change_90d_pct", "days_since_campaign_engagement",
            "rate_driven_transfers_12m", "referrals"]
X = StandardScaler().fit_transform(df[FEATURES])

# 2. Pick k - show the evidence, don't just assume 5
print("k  silhouette")
for k in range(2, 9):
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(X)
    print(f"{k}  {silhouette_score(X, labels):.3f}")

# 3. Final model
km = KMeans(5, n_init=20, random_state=42).fit(X)
df["cluster"] = km.labels_
centroids = pd.DataFrame(km.cluster_centers_, columns=FEATURES)

# 4. Name clusters by matching centroid profiles to business definitions
#    (positive weight = high is characteristic, negative = low is characteristic)
PROFILES = {
    "High-net-worth loyalist": {"log_balance": 1, "num_products": 1, "tenure_years": 1, "referrals": 1},
    "Wealth builder":          {"balance_change_6m_pct": 1, "log_balance": 0.5, "num_products": -0.3},
    "Silent churner":          {"login_change_90d_pct": -1, "balance_change_6m_pct": -1,
                                "days_since_campaign_engagement": 1, "logins_per_month": -1},
    "Rate shopper":            {"rate_driven_transfers_12m": 1.5},
    "Early-career client":     {"age": -1, "tenure_years": -1, "log_balance": -0.5},
}
score = np.array([[sum(w * centroids.loc[c, f] for f, w in prof.items()) for prof in PROFILES.values()]
                  for c in range(5)])
rows, cols = linear_sum_assignment(-score)       # best one-to-one cluster -> segment match
names = dict(zip(rows, [list(PROFILES)[j] for j in cols]))
df["segment"] = df["cluster"].map(names)

# 5. Action playbook - this is what makes segmentation useful
PLAYBOOK = {
    "High-net-worth loyalist": "White-glove relationship manager; private-banking events; referral rewards",
    "Wealth builder":          "Proactive advisor call this week; offer investment / managed portfolio product",
    "Silent churner":          "Personalised re-engagement now: RM outreach, fee review, 'we miss you' offer",
    "Rate shopper":            "Loyalty rate / tiered-yield incentive instead of generic newsletter",
    "Early-career client":     "Nurture: financial-wellness content, starter investment, first-home / credit journeys",
}
df["recommended_action"] = df["segment"].map(PLAYBOOK)

# 6. Business summary
summary = df.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_balance=("total_balance", "mean"),
    total_balance=("total_balance", "sum"),
    avg_products=("num_products", "mean"),
    avg_tenure=("tenure_years", "mean"),
    avg_balance_change_6m=("balance_change_6m_pct", "mean"),
    avg_login_change_90d=("login_change_90d_pct", "mean"),
    avg_days_idle=("days_since_campaign_engagement", "mean"),
).round(1)
summary["share_of_balances_%"] = (100 * summary["total_balance"] / summary["total_balance"].sum()).round(1)
print("\n", summary.to_string())

# Sanity check vs hidden archetypes (only possible because data is synthetic)
purity = (pd.crosstab(df["segment"], df["true_archetype"]).max(axis=1).sum() / len(df))
print(f"\nRecovery of hidden archetypes: {purity:.1%}")

# 7. Revenue-at-risk: balances held by silent churners
risk = df.loc[df.segment == "Silent churner", "total_balance"].sum()
print(f"Balances held by silent churners (at risk): ${risk:,.0f}")

df.drop(columns=["true_archetype"]).to_csv("outputs/customers_segmented.csv", index=False)
summary.to_csv("outputs/segment_summary.csv")

# 8. Charts
colors = dict(zip(PLAYBOOK, ["#1f4e79", "#2e8b57", "#c0392b", "#e69f00", "#7b5ea7"]))
pcs = PCA(2, random_state=42).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for s, c in colors.items():
    m = df.segment == s
    ax[0].scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=.5, c=c, label=s)
ax[0].set_title("Customer segments (PCA projection)"); ax[0].legend(markerscale=3, fontsize=8)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")

order = summary.sort_values("total_balance", ascending=False).index
share_c = (summary.loc[order, "customers"] / summary.customers.sum() * 100)
share_b = summary.loc[order, "share_of_balances_%"]
x = np.arange(len(order)); w = .38
ax[1].bar(x - w/2, share_c, w, label="% of customers", color="#9aa5b1")
ax[1].bar(x + w/2, share_b, w, label="% of balances", color="#1f4e79")
ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=20, ha="right", fontsize=8)
ax[1].set_title("Customers vs. money: not all segments are equal"); ax[1].legend()
plt.tight_layout(); plt.savefig("outputs/segments.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 4.5))
heat = df.groupby("segment")[FEATURES].mean()
heat = (heat - heat.mean()) / heat.std()
im = ax.imshow(heat.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index, fontsize=9)
ax.set_title("Segment profiles (z-scored feature means)"); plt.colorbar(im)
plt.tight_layout(); plt.savefig("outputs/segment_profiles.png", dpi=150)
print("\nSaved outputs/ (csvs + 2 charts)")
