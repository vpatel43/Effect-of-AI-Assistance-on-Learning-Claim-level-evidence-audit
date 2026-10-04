"""Step 1: coverage. What kinds of claims exist, about what, in which areas.

Counts are reported per *paper* as well as per claim: papers contribute 1-16
claims each, so claim counts overweight verbose abstracts.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 200)

df = pd.read_csv(OUT / "claims_clean.csv")
UNSTATED = {"not_stated", "not_applicable"}
FACETS = [
    "subject_type", "subject_mode", "object_type", "relation", "direction",
    "valence", "population_role", "population_setting", "measurement_horizon",
    "statistical_marking", "epistemic_strength",
]


def dual(idx_cols):
    """Claim count and unique-paper count for a grouping."""
    g = df.groupby(idx_cols)
    t = pd.DataFrame({"claims": g.size(), "papers": g.paper_id.nunique()})
    t["claims_per_paper"] = (t.claims / t.papers).round(2)
    return t.sort_values("papers", ascending=False)


def show(title, obj):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ---------------------------------------------------------------- marginals
for f in FACETS:
    show(f"MARGINAL  {f}", dual([f]))

# ------------------------------------------------------- the core 2x2 space
show("object_type x valence  (papers)",
     pd.crosstab(df.object_type, df.valence, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

show("object_type x direction  (papers)",
     pd.crosstab(df.object_type, df.direction, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

show("subject_type x object_type  (papers)",
     pd.crosstab(df.subject_type, df.object_type, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

show("population_setting x object_type  (papers)",
     pd.crosstab(df.population_setting, df.object_type, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

show("measurement_horizon x object_type  (papers)",
     pd.crosstab(df.measurement_horizon, df.object_type, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

show("measurement_horizon x valence  (papers)",
     pd.crosstab(df.measurement_horizon, df.valence, values=df.paper_id,
                 aggfunc=pd.Series.nunique).fillna(0).astype(int))

# ------------------------------------------------------------- cell density
# How much of the theoretically available cell space is actually occupied,
# and how concentrated is the occupancy.
def occupancy(cols):
    sizes = [df[c].nunique() for c in cols]
    total = int(np.prod(sizes))
    obs = df.groupby(cols).paper_id.nunique()
    obs = obs[obs > 0]
    n = len(obs)
    share = obs.sort_values(ascending=False)
    top10 = share.head(10).sum() / share.sum()
    # Papers needed to cover 50% / 90% of the mass
    cum = share.cumsum() / share.sum()
    return dict(
        facets="x".join(cols), possible=total, occupied=n,
        pct_occupied=round(100 * n / total, 1),
        cells_for_50pct=int((cum < .5).sum() + 1),
        cells_for_90pct=int((cum < .9).sum() + 1),
        top10_share=round(100 * top10, 1),
    )


occ = pd.DataFrame([
    occupancy(["subject_type", "object_type"]),
    occupancy(["relation", "direction", "valence"]),
    occupancy(["subject_type", "subject_mode", "object_type"]),
    occupancy(["object_type", "population_setting", "measurement_horizon"]),
    occupancy(["subject_type", "object_type", "relation", "direction"]),
    occupancy(FACETS),
])
show("CELL-SPACE OCCUPANCY", occ)

# ------------------------------------------------------- signature ranking
sig = dual(["signature"])
show("TOP 25 CLAIM SIGNATURES (by papers)", sig.head(25))
show("SIGNATURE TAIL", f"{(sig.papers == 1).sum()} signatures appear in exactly "
                       f"1 paper, out of {len(sig)} distinct signatures")

# --------------------------------------------------------- fully-specified
core = ["subject_type", "subject_mode", "object_type", "relation", "direction",
        "valence"]
full = df[~df[core].isin(UNSTATED).any(axis=1)]
show("FULLY-SPECIFIED CORE CLAIMS",
     f"{len(full)} claims ({100*len(full)/len(df):.1f}%) from "
     f"{full.paper_id.nunique()} papers have all 6 core facets stated")
show("...their signatures (top 20)",
     full.groupby("signature").agg(claims=("claim_id", "size"),
                                   papers=("paper_id", "nunique"))
         .sort_values("papers", ascending=False).head(20))

# ------------------------------------------------------------------ by year
show("valence x year (claim share within year)",
     (pd.crosstab(df.year, df.valence, normalize="index") * 100).round(1))
show("object_type x year (claim share within year)",
     (pd.crosstab(df.year, df.object_type, normalize="index") * 100).round(1))

occ.to_csv(OUT / "cell_occupancy.csv", index=False)
sig.to_csv(OUT / "signatures.csv")
