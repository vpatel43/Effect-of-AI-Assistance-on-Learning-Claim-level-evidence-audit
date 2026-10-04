"""Discover claim-level structure from the 19 fields, assuming no grouping.

Two unsupervised passes over the same encoding, all 19 fields entering on equal
terms:
  1. MCA  — what continuous axes organise the claims, and which fields drive
            each axis (field contributions are read off, not assumed).
  2. LCA  — how many discrete claim types the data supports, by BIC, and what
            each type looks like.

The one empirical decision: a category becomes its own column only if it occurs
in >= MIN_LEVEL claims. Rarer values are pooled into `<rare>` per field. This is
a frequency rule, applied identically to all 19 fields, not a semantic merge.
It exists because object_entity alone has 4576 levels and would otherwise carry
almost all the inertia.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

MIN_LEVEL = 30          # ~0.5% of 6063 claims

enc = pd.read_csv(OUT / "fields_encoded.csv")
FIELDS = list(enc.columns)
assert len(FIELDS) == 19
n = len(enc)


def show(title, obj):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ------------------------------------------------------- frequency recoding
rec = pd.DataFrame(index=enc.index)
for f in FIELDS:
    vc = enc[f].value_counts()
    keep = set(vc[vc >= MIN_LEVEL].index)
    rec[f] = np.where(enc[f].isin(keep), enc[f], "<rare>")
show("levels retained per field after the >=%d frequency rule" % MIN_LEVEL,
     pd.DataFrame({"levels_before": enc.nunique(), "levels_after": rec.nunique(),
                   "share_pooled_%": (rec == "<rare>").mean().mul(100).round(1)}))

Z = pd.get_dummies(rec, prefix_sep="=").astype(float)
p = len(FIELDS)
show("indicator matrix", f"{Z.shape[0]} claims x {Z.shape[1]} categories "
                         f"across {p} fields")

# ------------------------------------------------------------------ MCA
P = Z.values / Z.values.sum()
r = P.sum(1)
c = P.sum(0)
S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, sv, Vt = np.linalg.svd(S, full_matrices=False)
ev = sv ** 2

# Benzecri/Greenacre adjustment: raw MCA eigenvalues understate explained
# inertia because the indicator coding injects 1/p of noise per dimension.
thr = 1.0 / p
adj = np.array([((p / (p - 1)) * (l - thr)) ** 2 if l > thr else 0.0 for l in ev])
adj_pct = 100 * adj / adj.sum()

show("MCA dimensions", pd.DataFrame({
    "eigenvalue": ev[:12].round(4),
    "raw_%": (100 * ev[:12] / ev.sum()).round(1),
    "adjusted_%": adj_pct[:12].round(1),
    "cum_adjusted_%": adj_pct[:12].cumsum().round(1),
}, index=[f"dim{i+1}" for i in range(12)]))

# Column principal coordinates and contributions.
F = (Vt.T * sv) / np.sqrt(c)[:, None]
contrib = (c[:, None] * F ** 2) / ev            # category contribution to dim
cats = Z.columns.to_numpy()
cat_field = np.array([x.split("=")[0] for x in cats])

# Which FIELDS drive each axis -- summed over that field's categories.
fc = pd.DataFrame(contrib[:, :6], index=cat_field,
                  columns=[f"dim{i+1}" for i in range(6)])
fc = fc.groupby(level=0).sum().mul(100).round(1)
show("FIELD contribution to each MCA axis (%), summed over its categories",
     fc.sort_values("dim1", ascending=False))

for d in range(4):
    s = pd.Series(F[:, d], index=cats)
    ct = pd.Series(contrib[:, d], index=cats)
    top = ct.nlargest(12).index
    show(f"dim{d+1}: 12 highest-contributing categories "
         f"(adjusted {adj_pct[d]:.1f}% of inertia)",
         pd.DataFrame({"coord": s[top].round(2),
                       "contrib_%": (ct[top] * 100).round(1)})
           .sort_values("coord"))

# Claim coordinates, for downstream use.
G = (U * sv) / np.sqrt(r)[:, None]
np.save(OUT / "mca_claim_coords.npy", G[:, :10])


# ------------------------------------------------------------------ LCA
def lca(X, K, iters=300, tol=1e-6, seed=0):
    """EM for a latent class model over categorical fields."""
    rs = np.random.default_rng(seed)
    codes, sizes = [], []
    for j in range(X.shape[1]):
        u, inv = np.unique(X[:, j], return_inverse=True)
        codes.append(inv)
        sizes.append(len(u))
    N = X.shape[0]
    pi = rs.dirichlet(np.ones(K))
    theta = [rs.dirichlet(np.ones(s), size=K) for s in sizes]
    prev = -np.inf
    for _ in range(iters):
        logp = np.log(pi)[None, :].repeat(N, 0)
        for j, inv in enumerate(codes):
            logp += np.log(theta[j][:, inv].T + 1e-12)
        m = logp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))
        ll = lse.sum()
        R = np.exp(logp - lse[:, None])
        pi = R.mean(0)
        for j, inv in enumerate(codes):
            cnt = np.zeros((K, sizes[j]))
            np.add.at(cnt.T, inv, R)
            theta[j] = (cnt + 1e-3) / (cnt + 1e-3).sum(1, keepdims=True)
        if ll - prev < tol * abs(prev):
            break
        prev = ll
    npar = (K - 1) + K * sum(s - 1 for s in sizes)
    return dict(K=K, ll=ll, npar=npar, bic=-2 * ll + npar * np.log(N),
                aic=-2 * ll + 2 * npar, pi=pi, theta=theta, R=R,
                entropy=1 - (-(R * np.log(R + 1e-12)).sum() / (N * np.log(K))))


X = rec.values.astype(str)
fits = []
for K in range(2, 21):
    best = min((lca(X, K, seed=s) for s in range(3)), key=lambda d: d["bic"])
    fits.append(best)
    print(f"  LCA K={K:2d}  BIC={best['bic']:.0f}  AIC={best['aic']:.0f}  "
          f"entropy={best['entropy']:.3f}", flush=True)

tab = pd.DataFrame([{k: f[k] for k in ("K", "ll", "npar", "bic", "aic",
                                       "entropy")} for f in fits])
tab["bic_gain"] = -tab.bic.diff()      # improvement from adding one class
show("LCA model selection", tab.round(3).to_string(index=False))

# BIC keeps drifting down on large N with locally dependent items, so take the
# point where an extra class stops buying a material BIC improvement (<1% of
# the 2 -> 3 gain) rather than the raw minimum.
floor = 0.01 * tab.bic_gain.iloc[1]
elig = tab[(tab.bic_gain < floor) & (tab.K > 2)]
k_elbow = int(elig.K.iloc[0]) if len(elig) else int(tab.K.iloc[-1])
show("selection", f"raw BIC minimum: K={int(tab.loc[tab.bic.idxmin(),'K'])}\n"
                  f"elbow (BIC gain < 1% of first gain): K={k_elbow}\n"
                  f"profiling K={k_elbow}")
best = [f for f in fits if f["K"] == k_elbow][0]

assign = best["R"].argmax(1)
rec["class"] = assign
show("class sizes", pd.Series(assign).value_counts().sort_index()
     .rename("claims").to_frame().assign(
         share_pct=lambda d: (100 * d.claims / n).round(1)))

# Profile each class: for every field, the levels most over-represented
# relative to the corpus baseline.
for k in range(best["K"]):
    g = rec[rec["class"] == k]
    rows = []
    for f in FIELDS:
        base = rec[f].value_counts(normalize=True)
        got = g[f].value_counts(normalize=True)
        lift = (got / base.reindex(got.index)).dropna()
        top = lift[got >= .15].nlargest(2)
        for lvl, lf in top.items():
            rows.append(dict(field=f, level=lvl, in_class=round(100*got[lvl], 1),
                             corpus=round(100*base[lvl], 1), lift=round(lf, 2)))
    prof = pd.DataFrame(rows).sort_values("lift", ascending=False)
    show(f"CLASS {k}  (n={len(g)}, {100*len(g)/n:.1f}%)  "
         f"— levels >=15% within class, ranked by lift",
         prof.head(14).to_string(index=False))

rec.to_csv(OUT / "claims_lca.csv", index=False)
tab.to_csv(OUT / "lca_selection.csv", index=False)
fc.to_csv(OUT / "mca_field_contributions.csv")
