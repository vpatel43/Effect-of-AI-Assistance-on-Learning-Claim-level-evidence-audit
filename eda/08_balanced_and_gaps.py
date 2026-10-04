"""What the corpus looks like on a balanced evidence base, and where the gaps
hide.

Part A -- standardisation. Three views of the same corpus:
    raw       one row per claim, as extracted
    paper-wt  each paper contributes weight 1, split across its claims, so a
              16-claim abstract stops outvoting sixteen studies
    balanced  paper-weighted, then reweighted so taxonomic, directional and
              magnitude claims (claim specificity) each carry a third of the total

Part B -- hidden gaps. A gap is "hidden" when the margins promise coverage the
joint does not deliver: cell expected to hold E papers under independence of
its margins, observed to hold far fewer. Also flagged: cells that look covered
but rest on one paper, and cells whose entire evidence base is taxonomic or
carries no statistical marking.

Axis fields are the ones MCA showed carry the inertia, not a chosen scheme.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 200)

raw = pd.read_csv(BASE / "claims_with_semantics.csv")
enc = pd.read_csv(OUT / "fields_encoded.csv")
df = pd.concat([raw[["claim_id", "paper_id", "year", "evidence_type",
                     "work_type"]], enc], axis=1)

SENT = {"not stated", "not applicable", "missing", "nan", "none", "<rare>"}


def show(title, obj):
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    print(obj if isinstance(obj, str) else obj.to_string())


def stated(col):
    return ~df[col].isin(SENT)


# ============================================================ PART A weights
df["w_raw"] = 1.0
df["w_paper"] = 1.0 / df.groupby("paper_id").claim_id.transform("size")

# Balance evidence_type on top of the paper weighting.
share = df.groupby("evidence_type").w_paper.sum()
share = share / share.sum()
target = pd.Series({"D": 1/3, "M": 1/3, "T": 1/3})
df["w_bal"] = df.w_paper * df.evidence_type.map(target / share)

show("weighting schemes",
     pd.DataFrame({
         "total weight": [df.w_raw.sum(), df.w_paper.sum(), df.w_bal.sum()],
     }, index=["raw", "paper-wt", "balanced"]).round(1).to_string()
     + "\n\nevidence_type share of total weight:\n"
     + pd.DataFrame({
         "raw": df.groupby("evidence_type").w_raw.sum() / df.w_raw.sum(),
         "paper-wt": df.groupby("evidence_type").w_paper.sum() / df.w_paper.sum(),
         "balanced": df.groupby("evidence_type").w_bal.sum() / df.w_bal.sum(),
     }).mul(100).round(1).to_string())


def wdist(col, restrict_stated=True):
    """Weighted level shares of one field under each scheme."""
    d = df[stated(col)] if restrict_stated else df
    out = {}
    for w in ["w_raw", "w_paper", "w_bal"]:
        s = d.groupby(col)[w].sum()
        out[w] = 100 * s / s.sum()
    t = pd.DataFrame(out).rename(columns={"w_raw": "raw_%",
                                          "w_paper": "paper_wt_%",
                                          "w_bal": "balanced_%"})
    t["shift_pp"] = t["balanced_%"] - t["raw_%"]
    return t.sort_values("balanced_%", ascending=False).round(1)


for f in ["valence", "relation", "direction", "object_type", "subject_type",
          "subject_mode", "population_setting", "population_role",
          "measurement_horizon", "statistical_marking", "epistemic_strength"]:
    show(f"STANDARDISED — {f} (stated claims only)", wdist(f))

# Reporting completeness is itself an outcome worth standardising.
comp = pd.DataFrame({
    lbl: [100 * df.loc[stated(c), w].sum() / df[w].sum() for c in
          ["relation", "direction", "valence", "object_type",
           "population_setting", "measurement_horizon", "statistical_marking"]]
    for lbl, w in [("raw_%", "w_raw"), ("paper_wt_%", "w_paper"),
                   ("balanced_%", "w_bal")]
}, index=["relation", "direction", "valence", "object_type",
          "population_setting", "measurement_horizon", "statistical_marking"])
comp["shift_pp"] = comp["balanced_%"] - comp["raw_%"]
show("STANDARDISED — share of claims with each field stated", comp.round(1))


# ============================================================ PART B  gaps
AX = ["subject_type", "object_type", "population_setting"]
sub = df[stated(AX[0]) & stated(AX[1]) & stated(AX[2])].copy()
show("gap analysis sample",
     f"{len(sub)} claims ({100*len(sub)/len(df):.1f}%) from "
     f"{sub.paper_id.nunique()} papers state all three of "
     f"{', '.join(AX)}\n"
     f"cell space: {sub[AX[0]].nunique()} x {sub[AX[1]].nunique()} x "
     f"{sub[AX[2]].nunique()} = "
     f"{np.prod([sub[a].nunique() for a in AX])} cells")

# Observed vs expected under independence of the three margins.
N = len(sub)
marg = {a: sub[a].value_counts(normalize=True) for a in AX}
idx = pd.MultiIndex.from_product([marg[a].index for a in AX], names=AX)
exp = pd.Series(
    [N * marg[AX[0]][i] * marg[AX[1]][j] * marg[AX[2]][k] for i, j, k in idx],
    index=idx, name="expected_claims")

obs = sub.groupby(AX).size().reindex(idx, fill_value=0).rename("obs_claims")
pap = sub.groupby(AX).paper_id.nunique().reindex(idx, fill_value=0).rename("obs_papers")
tab = pd.concat([obs, pap, exp.round(1)], axis=1)
tab["ratio"] = (tab.obs_claims / tab.expected_claims).round(2)

# Evidence composition inside each cell.
ev = sub.groupby(AX).evidence_type.value_counts().unstack(fill_value=0) \
        .reindex(idx, fill_value=0)
tab["pct_directional"] = (100 * ev.get("D", 0) / obs.replace(0, np.nan)).round(0)
tab["pct_taxonomic"] = (100 * ev.get("T", 0) / obs.replace(0, np.nan)).round(0)
tab["n_sig_marked"] = sub[sub.statistical_marking.eq("significant")] \
    .groupby(AX).size().reindex(idx, fill_value=0)

show("HIDDEN GAPS — expected >= 3 claims, observed/expected <= 0.4, ranked by "
     "the shortfall (expected - observed)",
     tab.assign(shortfall=(tab.expected_claims - tab.obs_claims).round(1))
        [(tab.expected_claims >= 3) & (tab.ratio <= 0.4)]
        .sort_values("shortfall", ascending=False).head(25))

show("EMPTY CELLS with the largest marginal promise (observed 0)",
     tab[(tab.obs_claims == 0) & (tab.expected_claims >= 1.5)]
        .sort_values("expected_claims", ascending=False).head(25))

show("THIN CELLS — <=2 papers but non-trivial marginal promise (expected >= 3)",
     tab[(tab.obs_papers <= 2) & (tab.expected_claims >= 3)]
        .sort_values("expected_claims", ascending=False).head(20))

show("EVIDENCE-THIN CELLS — >=8 claims, >=3 papers, but 0 significance-marked",
     tab[(tab.obs_claims >= 8) & (tab.obs_papers >= 3) & (tab.n_sig_marked == 0)]
        .sort_values("obs_claims", ascending=False).head(20))

show("TAXONOMIC-HEAVY CELLS — >=8 claims, >=40% taxonomic claims",
     tab[(tab.obs_claims >= 8) & (tab.pct_taxonomic >= 40)]
        .sort_values("obs_claims", ascending=False).head(20))

show("the populated core, for contrast (top 20 cells by papers)",
     tab.sort_values("obs_papers", ascending=False).head(20))

# Two-way version on the axes most likely to matter for a learning argument:
# what is measured, when, and on whom -- reported in papers, not claims.
for pair in [("object_type", "measurement_horizon"),
             ("object_type", "population_role"),
             ("subject_type", "measurement_horizon"),
             ("population_setting", "measurement_horizon")]:
    s2 = df[stated(pair[0]) & stated(pair[1])]
    ct = pd.crosstab(s2[pair[0]], s2[pair[1]], values=s2.paper_id,
                     aggfunc=pd.Series.nunique).fillna(0).astype(int)
    n2 = len(s2)
    m0 = s2[pair[0]].value_counts(normalize=True)
    m1 = s2[pair[1]].value_counts(normalize=True)
    e2 = pd.DataFrame(np.outer(m0[ct.index], m1[ct.columns]) * n2,
                      index=ct.index, columns=ct.columns)
    oc = pd.crosstab(s2[pair[0]], s2[pair[1]])
    show(f"{pair[0]} x {pair[1]} — unique papers", ct)
    show(f"{pair[0]} x {pair[1]} — observed/expected claims (<0.5 = hidden gap)",
         (oc / e2).round(2))


# ------------------------------------------------------------- the key cube
# what is measured x on whom x over what horizon. This is the joint that a
# learning argument needs and that no margin reveals.
CUBE = ["object_type", "population_role", "measurement_horizon"]
c3 = df[stated(CUBE[0]) & stated(CUBE[1]) & stated(CUBE[2])].copy()
show("horizon cube sample",
     f"{len(c3)} claims ({100*len(c3)/len(df):.1f}%) from "
     f"{c3.paper_id.nunique()} papers state outcome, population AND horizon")

cm = {a: c3[a].value_counts(normalize=True) for a in CUBE}
ci = pd.MultiIndex.from_product([cm[a].index for a in CUBE], names=CUBE)
ce = pd.Series([len(c3) * cm[CUBE[0]][i] * cm[CUBE[1]][j] * cm[CUBE[2]][k]
                for i, j, k in ci], index=ci).round(1)
co = c3.groupby(CUBE).size().reindex(ci, fill_value=0)
cp = c3.groupby(CUBE).paper_id.nunique().reindex(ci, fill_value=0)
cube = pd.DataFrame({"obs_claims": co, "obs_papers": cp, "expected": ce})
cube["ratio"] = (cube.obs_claims / cube.expected).round(2)

show("horizon cube — every cell with expected >= 2, sorted by shortfall",
     cube.assign(shortfall=(cube.expected - cube.obs_claims).round(1))
         [cube.expected >= 2].sort_values("shortfall", ascending=False).head(30))
show("horizon cube — what IS covered (>= 3 papers)",
     cube[cube.obs_papers >= 3].sort_values("obs_papers", ascending=False))

tab.to_csv(OUT / "gap_cells.csv")
cube.to_csv(OUT / "gap_cube_horizon.csv")
