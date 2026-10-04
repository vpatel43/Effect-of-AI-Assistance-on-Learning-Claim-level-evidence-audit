"""Why are benefits over-represented and harms under-represented?

Distinguishes the candidate explanations that the corpus can actually separate:
  A. Underpowered  -- harms were tested and came out non-significant
  B. Untested      -- harms were asserted without any test being reported
  C. Undermeasured -- the outcomes and horizons on which harm appears are
                      rarely measured at all
  D. Compositional -- harms arrive disproportionately as taxonomic (less specific) claims
  E. Definitional  -- valence is largely entailed by direction, so what counts
                      as harm depends on which outcomes were chosen
  F. Secondary     -- harms are ancillary findings rather than headline results
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 220)

raw = pd.read_csv(BASE / "claims_with_semantics.csv")
enc = pd.read_csv(OUT / "fields_encoded.csv")
meta = raw[["claim_id", "paper_id", "evidence_type", "claim_role",
            "n_reported"]].rename(columns={"n_reported": "n_raw"})
df = pd.concat([meta, enc], axis=1)
# Claim index within the paper: c1 is the first claim the extractor emitted.
df["claim_ix"] = raw.claim_id.str.extract(r"::c(\d+)$").astype(float)

d = df[df.valence.isin(["beneficial", "harmful"])].copy()
d["harm"] = (d.valence == "harmful").astype(int)


def show(title, obj):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ---------------------------------------------------- A vs B: tested or not?
show("statistical_marking by valence (claims)",
     pd.crosstab(d.valence, d.statistical_marking))
show("...row %", (pd.crosstab(d.valence, d.statistical_marking,
                              normalize="index") * 100).round(1))

tested = d[d.statistical_marking.isin(["significant", "not significant"])]
show("AMONG CLAIMS THAT REPORT A TEST: was it significant?",
     pd.crosstab(tested.valence, tested.statistical_marking,
                 normalize="index").mul(100).round(1))
show("...n", pd.crosstab(tested.valence, tested.statistical_marking))

if tested.harm.nunique() > 1:
    tested = tested.copy()
    tested["ns"] = (tested.statistical_marking == "not significant").astype(int)
    m = smf.logit("ns ~ harm", data=tested).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": tested.paper_id})
    show("odds a *reported test* is non-significant, harm vs benefit",
         f"OR = {np.exp(m.params['harm']):.2f}   p = {m.pvalues['harm']:.3f}   "
         f"n = {len(tested)} claims, {tested.paper_id.nunique()} papers")

# ------------------------------------------- can power be assessed directly?
show("n_reported availability",
     f"claims with a sample size: {int(df.n_raw.notna().sum())} of {len(df)} "
     f"({100*df.n_raw.notna().mean():.1f}%)\n"
     f"beneficial-or-harmful claims with a sample size: "
     f"{int(d.n_raw.notna().sum())}\n"
     f"  of those, harmful: {int(d.loc[d.n_raw.notna(), 'harm'].sum())}\n"
     f"median n where reported: "
     f"{df.n_raw.median() if df.n_raw.notna().any() else 'n/a'}")

# ------------------------------- C: is harm concentrated in rare outcomes?
tab = pd.DataFrame({
    "claims": d.groupby("object_type").size(),
    "corpus_share_%": (100 * df.groupby("object_type").size() / len(df)).round(1),
    "harm_%": (100 * d.groupby("object_type").harm.mean()).round(1),
}).sort_values("harm_%", ascending=False)
show("harm rate vs how often the outcome type is measured at all", tab)

hz = pd.DataFrame({
    "claims": d.groupby("measurement_horizon").size(),
    "corpus_share_%": (100 * df.groupby("measurement_horizon").size()
                       / len(df)).round(1),
    "harm_%": (100 * d.groupby("measurement_horizon").harm.mean()).round(1),
})
show("harm rate vs how often the horizon is measured at all", hz)


def standardise(strat_col, levels=None):
    """Harm rate if the stratifying variable were uniformly distributed."""
    g = d[d[strat_col].isin(levels)] if levels else d
    rates = g.groupby(strat_col).harm.mean()
    obs = g.groupby(strat_col).size() / len(g)
    return dict(variable=strat_col,
                observed_harm_pct=round(100 * float((rates * obs).sum()), 1),
                balanced_harm_pct=round(100 * float(rates.mean()), 1),
                strata=len(rates))


SUBSTANTIVE_OBJ = ["task performance", "cognitive process", "behavioural strategy",
                   "disposition or attitude", "subjective experience"]
SUBSTANTIVE_HZ = ["immediate", "delayed", "longitudinal"]
show("harm rate under a balanced design mix",
     pd.DataFrame([
         standardise("object_type", SUBSTANTIVE_OBJ),
         standardise("measurement_horizon", SUBSTANTIVE_HZ),
         standardise("evidence_type"),
         standardise("population_setting",
                     ["education", "clinical", "laboratory", "workplace"]),
     ]))

# --------------------------------------------- E: is valence just direction?
show("valence x direction (row % within direction)",
     pd.crosstab(df.direction, df.valence, normalize="index").mul(100).round(1))
show("...how often does an increase get called harmful?",
     f"increase -> harmful: "
     f"{100*((df.direction=='increase') & (df.valence=='harmful')).sum() / max((df.direction=='increase').sum(),1):.1f}%\n"
     f"decrease -> harmful: "
     f"{100*((df.direction=='decrease') & (df.valence=='harmful')).sum() / max((df.direction=='decrease').sum(),1):.1f}%")

# -------------------------------------------------- F: are harms secondary?
show("harm rate by claim position within the paper",
     d.groupby(d.claim_ix.clip(upper=6)).agg(
         claims=("harm", "size"), harm_pct=("harm", lambda s: round(100*s.mean(), 1))))
sub = d[d.claim_ix.notna()].copy()
m = smf.logit("harm ~ claim_ix", data=sub).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": sub.paper_id})
show("odds of harm per additional position down the claim list",
     f"OR = {np.exp(m.params['claim_ix']):.3f}  p = {m.pvalues['claim_ix']:.2e}")

show("harm rate by claim_role",
     d.groupby("claim_role").agg(claims=("harm", "size"),
                                 harm_pct=("harm", lambda s: round(100*s.mean(), 1))))
