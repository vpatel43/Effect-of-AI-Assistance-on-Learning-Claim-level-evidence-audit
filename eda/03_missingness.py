"""Step 2: treat `not_stated` as data, and separate instrument from literature.

Every claim in this corpus was extracted from an abstract (retrieval_depth is
constant), by one of three extractors over one of three channels. Before any
facet distribution is read as a fact about the literature, it has to survive
the question: is this the abstracts, or is it the pipeline?
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 220)

df = pd.read_csv(OUT / "claims_clean.csv")
FACETS = [
    "subject_type", "subject_mode", "object_type", "relation", "direction",
    "valence", "population_role", "population_setting", "measurement_horizon",
    "statistical_marking", "epistemic_strength",
]
UNSTATED = {"not_stated", "not_applicable"}


def show(title, obj):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ------------------------------------------------ is the pipeline confounded?
show("extractor x channel (claims)", pd.crosstab(df.extractor, df.channel))
show("extractor x year (row %)",
     (pd.crosstab(df.extractor, df.year, normalize="index") * 100).round(1))
show("extractor x work_type_c (row %)",
     (pd.crosstab(df.extractor, df.work_type_c, normalize="index") * 100).round(1))
show("extractor x evidence_type (row %)",
     (pd.crosstab(df.extractor, df.evidence_type, normalize="index") * 100).round(1))
show("extractor x claim_role (row %)",
     (pd.crosstab(df.extractor, df.claim_role, normalize="index") * 100).round(1))

# Facet-by-facet: how much does the label distribution move across extractors?
# Total variation distance between each extractor's distribution and the pooled
# one; chi2 for reference (inflated by clustering, so read the TVD).
rows = []
for f in FACETS + ["core_shape"]:
    pooled = df[f].value_counts(normalize=True)
    for src_col in ["extractor", "channel"]:
        for lvl, g in df.groupby(src_col):
            p = g[f].value_counts(normalize=True).reindex(pooled.index).fillna(0)
            rows.append(dict(facet=f, source=src_col, level=lvl, n=len(g),
                             tvd=round(0.5 * np.abs(p - pooled).sum(), 3)))
inst = pd.DataFrame(rows).pivot_table(index="facet", columns=["source", "level"],
                                      values="tvd")
show("INSTRUMENT EFFECT: total-variation distance from pooled distribution",
     inst.round(3))
show("worst facets by max TVD across extractors",
     inst["extractor"].max(axis=1).sort_values(ascending=False).round(3))

# --------------------------------------------- what predicts being *stated*?
df["year_c"] = df.year - 2023
# epistemic_strength is 99.5% stated -- a logit on it is separated, not
# informative. Drop it and report its rate descriptively instead.
MODELLED = [f for f in FACETS if f != "epistemic_strength"]
res = []
for f in MODELLED:
    df["_y"] = (~df[f].isin(UNSTATED)).astype(int)
    if df._y.nunique() < 2:
        continue
    m = smf.logit(
        "_y ~ C(evidence_type, Treatment('D')) + C(claim_role) "
        "+ C(work_type_c, Treatment('article')) + year_c",
        data=df,
    ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df.paper_id})
    for term in m.params.index:
        if term == "Intercept":
            continue
        beta = m.params[term]
        # |beta| > 5 with a tiny cell means quasi-separation, not a real effect.
        separated = abs(beta) > 5 or not m.mle_retvals["converged"]
        res.append(dict(facet=f, term=term,
                        odds_ratio=np.nan if separated else round(np.exp(beta), 2),
                        p=np.nan if separated else round(m.pvalues[term], 4),
                        separated=separated))
res = pd.DataFrame(res)
show("P(facet stated) — odds ratios, paper-clustered SEs "
     "(blank = quasi-separated, not estimable)",
     res.pivot(index="term", columns="facet", values="odds_ratio").fillna(""))
show("...same, p<.05 only",
     res[res.p < .05].pivot(index="term", columns="facet",
                            values="odds_ratio").fillna(""))
show("dropped for separation", res[res.separated][["facet", "term"]].to_string(index=False))
show("epistemic_strength (descriptive, not modelled)",
     df.groupby("evidence_type").epistemic_strength.value_counts(normalize=True)
       .mul(100).round(1))

# ------------------------------------------------- co-missingness structure
mask = pd.DataFrame({c: (~df[c].isin(UNSTATED)).astype(int) for c in FACETS})
show("tetrachoric-ish: correlation of stated-ness across facets",
     mask.corr().round(2))

# Does unstated-ness travel with the claim or with the paper?
pv = df.groupby("paper_id").n_stated.agg(["mean", "std", "size"])
within = pv[pv["size"] > 1]["std"].mean()
between = pv["mean"].std()
show("variance decomposition of n_stated",
     f"between-paper SD of paper mean: {between:.2f}\n"
     f"mean within-paper SD:            {within:.2f}\n"
     f"=> {'paper-level' if between > within else 'claim-level'} reporting "
     f"drives how much gets stated")

res.to_csv(OUT / "missingness_models.csv", index=False)
inst.to_csv(OUT / "instrument_tvd.csv")
