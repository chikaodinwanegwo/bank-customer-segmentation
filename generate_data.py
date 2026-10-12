"""Generate a synthetic bank customer base (no real customer data needed)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (share of base, feature generators) - hidden archetypes the model must rediscover
ARCHETYPES = {
    "hnw_loyalist": dict(n=0.10, age=(55, 8), tenure=(14, 4), balance=(13.5, 0.5), products=(5.5, 0.9),
                         bal_chg=(2, 3), logins=(14, 4), login_chg=(0, 8), days_idle=(10, 6),
                         rate_moves=(0.3, 0.5), referrals=(3.5, 1.2)),
    "wealth_builder": dict(n=0.20, age=(38, 7), tenure=(6, 3), balance=(11.2, 0.4), products=(2.2, 0.7),
                           bal_chg=(18, 6), logins=(12, 4), login_chg=(5, 8), days_idle=(20, 10),
                           rate_moves=(0.5, 0.7), referrals=(0.6, 0.8)),
    "silent_churner": dict(n=0.15, age=(45, 12), tenure=(5, 3), balance=(9.8, 0.8), products=(1.5, 0.6),
                           bal_chg=(-25, 8), logins=(2, 1.5), login_chg=(-60, 15), days_idle=(120, 30),
                           rate_moves=(0.4, 0.6), referrals=(0.1, 0.3)),
    "rate_shopper": dict(n=0.25, age=(42, 11), tenure=(3, 2), balance=(10.6, 0.6), products=(1.4, 0.5),
                         bal_chg=(0, 25), logins=(9, 3), login_chg=(5, 15), days_idle=(35, 20),
                         rate_moves=(5, 1.5), referrals=(0.2, 0.4)),
    "early_career": dict(n=0.30, age=(26, 3), tenure=(1.5, 1), balance=(7.8, 0.6), products=(1.3, 0.5),
                         bal_chg=(8, 10), logins=(18, 5), login_chg=(3, 10), days_idle=(25, 12),
                         rate_moves=(0.4, 0.6), referrals=(0.3, 0.5)),
}

rows = []
N = 5000
for name, p in ARCHETYPES.items():
    n = int(N * p["n"])
    rows.append(pd.DataFrame({
        "true_archetype": name,
        "age": rng.normal(*p["age"], n).clip(21, 85),
        "tenure_years": rng.normal(*p["tenure"], n).clip(0.1, 40),
        "total_balance": np.exp(rng.normal(*p["balance"], n)),          # log-normal balances
        "num_products": rng.normal(*p["products"], n).round().clip(1, 8),
        "balance_change_6m_pct": rng.normal(*p["bal_chg"], n),
        "logins_per_month": rng.normal(*p["logins"], n).clip(0, 60),
        "login_change_90d_pct": rng.normal(*p["login_chg"], n).clip(-100, 100),
        "days_since_campaign_engagement": rng.normal(*p["days_idle"], n).clip(0, 365),
        "rate_driven_transfers_12m": rng.normal(*p["rate_moves"], n).round().clip(0, 12),
        "referrals": rng.normal(*p["referrals"], n).round().clip(0, 10),
    }))

df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.insert(0, "customer_id", [f"C{100000+i}" for i in range(len(df))])
df.round(2).to_csv("data/customers.csv", index=False)
print(f"Saved {len(df):,} synthetic customers -> data/customers.csv")
