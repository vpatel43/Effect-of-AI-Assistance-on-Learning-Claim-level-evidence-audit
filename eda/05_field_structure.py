"""Discover structure among the 19 semantic fields, assuming none.

Every one of the 19 fields is treated identically: as a labelling of the 6063
claims. Association between fields is adjusted mutual information, which
corrects for chance agreement and so does not reward high-cardinality fields.
Fields are then clustered on 1 - AMI. Whatever groups appear, appear.

The only preprocessing is mechanical string folding (case, underscores,
hyphens, whitespace) so that `non_AI_external_aid` and `non-AI external aid`
are not counted as two categories. No values are merged on meaning.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score as ami
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

raw = pd.read_csv(BASE / "claims_with_semantics.csv")
FIELDS = list(raw.columns[18:37])          # subject_type .. evidence_span
assert len(FIELDS) == 19

# Pipeline sentinels: strings the extractor emits to mean "no value", as
# opposed to a substantive label. Identifiable without semantic judgement.
SENTINEL = {"not stated", "not applicable", "none", "", "nan", "missing"}


def fold(s):
    """Mechanical string hygiene only."""
    return (s.astype(str).str.strip().str.lower()
             .str.replace(r"[_\-]+", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True))


def show(title, obj):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ------------------------------------------------------------------- encode
enc = pd.DataFrame(index=raw.index)
for f in FIELDS:
    col = raw[f]
    # magnitude_value reads as object because a single row holds the string
    # "not_stated"; coerce so it is treated as the numeric field it is.
    num = pd.to_numeric(col, errors="coerce")
    if col.notna().sum() and num.notna().sum() / max(col.notna().sum(), 1) > .95:
        col = num
    if col.dtype.kind in "if":
        # Numeric: keep NaN as its own label, bin the rest by observed deciles
        # so the field can enter an information-theoretic comparison at all.
        lab = pd.Series("missing", index=raw.index, dtype=object)
        obs = col.notna()
        if obs.sum() > 10:
            lab[obs] = pd.qcut(col[obs], 10, duplicates="drop").astype(str)
        enc[f] = lab
    else:
        enc[f] = fold(col).fillna("missing")

card = pd.DataFrame({
    "levels": enc.nunique(),
    "modal_share_%": (enc.apply(lambda c: c.value_counts(normalize=True).iloc[0])
                      * 100).round(1),
    "sentinel_%": (enc.apply(lambda c: c.isin(SENTINEL | {"missing"}).mean())
                   * 100).round(1),
}).loc[FIELDS]
show("THE 19 FIELDS AS THE DATA PRESENTS THEM", card)

# --------------------------------------------------- pairwise field structure
A = pd.DataFrame(np.eye(19), index=FIELDS, columns=FIELDS)
for i, a in enumerate(FIELDS):
    for b in FIELDS[i + 1:]:
        v = ami(enc[a], enc[b], average_method="arithmetic")
        A.loc[a, b] = A.loc[b, a] = round(v, 3)
show("ADJUSTED MUTUAL INFORMATION between all 19 fields", A)

pairs = (A.where(np.triu(np.ones(A.shape), 1).astype(bool))
          .stack().sort_values(ascending=False))
show("strongest 25 field pairs", pairs.head(25))
show("weakest 15 field pairs", pairs.tail(15))

# ------------------------------------------------------- cluster the fields
D = (1 - A).clip(lower=0).values
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
for k in (2, 3, 4, 5, 6, 7):
    lab = fcluster(Z, k, criterion="maxclust")
    groups = pd.Series(lab, index=FIELDS).groupby(lambda x: None).apply(lambda s: s)
    out = {}
    for f, g in zip(FIELDS, lab):
        out.setdefault(g, []).append(f)
    show(f"FIELD CLUSTERS at k={k}",
         "\n".join(f"  G{g}: " + ", ".join(v) for g, v in sorted(out.items())))

# Merge order, so the grouping can be read as a hierarchy rather than a cut.
dn = dendrogram(Z, labels=FIELDS, no_plot=True)
show("field order along the dendrogram", " -> ".join(dn["ivl"]))
merges = []
names = {i: f for i, f in enumerate(FIELDS)}
for i, (a, b, d, n) in enumerate(Z):
    names[19 + i] = f"({names[int(a)]} + {names[int(b)]})"
    merges.append(dict(step=i + 1, distance=round(d, 3), size=int(n),
                       merged=names[19 + i]))
show("merge sequence (1 - AMI)", pd.DataFrame(merges))

# ------------------------- is the association structure just shared silence?
# Same computation on stated/not-stated indicators alone.
ind = enc.apply(lambda c: (~c.isin(SENTINEL | {"missing"})).astype(int))
keep = [f for f in FIELDS if ind[f].nunique() > 1]
B = pd.DataFrame(np.eye(len(keep)), index=keep, columns=keep)
for i, a in enumerate(keep):
    for b in keep[i + 1:]:
        v = ami(ind[a], ind[b], average_method="arithmetic")
        B.loc[a, b] = B.loc[b, a] = round(v, 3)
show("AMI between STATED-NESS indicators only (is the structure just silence?)",
     B)

# Value structure with silence removed: recompute AMI on claims where both
# fields carry a substantive value.
C = pd.DataFrame(np.eye(19), index=FIELDS, columns=FIELDS)
N = pd.DataFrame(np.zeros((19, 19)), index=FIELDS, columns=FIELDS)
for i, a in enumerate(FIELDS):
    for b in FIELDS[i + 1:]:
        m = ind[a].astype(bool) & ind[b].astype(bool) if a in keep and b in keep \
            else pd.Series(False, index=enc.index)
        N.loc[a, b] = N.loc[b, a] = m.sum()
        if m.sum() > 200:
            v = ami(enc.loc[m, a], enc.loc[m, b], average_method="arithmetic")
        else:
            v = np.nan
        C.loc[a, b] = C.loc[b, a] = round(v, 3) if v == v else np.nan
show("AMI on co-stated claims only (silence removed)", C)
show("...n of co-stated claims per pair", N.astype(int))

delta = (A - C).round(3)
show("AMI drop when silence is removed (positive = association was silence)",
     delta)

A.to_csv(OUT / "field_ami.csv")
C.to_csv(OUT / "field_ami_costated.csv")
B.to_csv(OUT / "field_ami_statedness.csv")
enc.to_csv(OUT / "fields_encoded.csv", index=False)
