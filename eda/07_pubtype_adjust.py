"""Adjusting the harm/benefit evidential asymmetry for publication type.

Four adjustments of increasing strength, on the same four outcomes:
  M0  unadjusted
  M1  + evidence_type          (claim specificity: taxonomic / directional / magnitude)
  M2  + evidence_type + format (paper-level: article / preprint / conference / review)
  M3  paper fixed effects      (conditional logit; absorbs everything about the
                                paper, including publication type, venue, year,
                                topic and author)
Plus stratified estimates and a post-stratified (standardised) estimate for the
question "what would the asymmetry look like if evidence types were balanced".
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 220)

raw = pd.read_csv(BASE / "claims_with_semantics.csv")
enc = pd.read_csv(OUT / "fields_encoded.csv")
df = pd.concat([raw[["claim_id", "paper_id", "year", "venue", "work_type",
                     "evidence_type", "claim_role"]], enc], axis=1)


def show(title, obj):
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ------------------------------------------ split format from study design
# work_type mixes two vocabularies: Crossref emits a bare format label,
# PubMed emits a semicolon list of publication types that includes design.
wt = df.work_type.fillna("unknown").str.lower()
df["format"] = np.select(
    [wt.str.contains("preprint"), wt.str.contains("conference"),
     wt.str.contains("report"), wt.eq("review"),
     wt.str.contains("article|journal")],
    ["preprint", "conference", "report", "review_crossref", "article"],
    default="unknown")
df["design"] = np.select(
    [wt.str.contains("randomized|randomised"),
     wt.str.contains("meta-analysis"),
     wt.str.contains("systematic review|scoping review"),
     wt.str.contains("observational|comparative study"),
     wt.str.contains("; review"), wt.eq("review")],
    ["rct", "meta_analysis", "systematic_review", "observational",
     "narrative_review", "narrative_review"],
    default="not_reported")

show("how much design information work_type actually carries",
     df.groupby("design").agg(claims=("claim_id", "size"),
                              papers=("paper_id", "nunique"))
       .sort_values("papers", ascending=False))
show("format", df.groupby("format").agg(claims=("claim_id", "size"),
                                        papers=("paper_id", "nunique"))
     .sort_values("papers", ascending=False))
show("evidence_type — the one evidential variable that is complete",
     df.groupby("evidence_type").agg(claims=("claim_id", "size"),
                                     papers=("paper_id", "nunique")))

# ------------------------------------------------------------ the outcomes
d = df[df.valence.isin(["beneficial", "harmful"])].copy()
d["harm"] = (d.valence == "harmful").astype(int)
OUTCOMES = {
    "associative": d.relation.eq("associates"),
    "hedged_or_speculative": d.epistemic_strength.isin(["hedged", "speculative"]),
    "significance_marked": d.statistical_marking.eq("significant"),
    "carries_magnitude": d.magnitude_value.ne("missing"),
}

show("analysis sample",
     f"{len(d)} beneficial-or-harmful claims from {d.paper_id.nunique()} papers\n"
     f"papers containing BOTH valences (usable by the fixed-effects model): "
     f"{(d.groupby('paper_id').valence.nunique() == 2).sum()}")

# ----------------------------------------------------- composition check
show("is publication type associated with valence at all? (row % harmful)",
     pd.concat([
         d.groupby("evidence_type").harm.mean().mul(100).rename("harm_%"),
         d.groupby("evidence_type").size().rename("n"),
     ], axis=1).round(1))
show("...by format",
     pd.concat([d.groupby("format").harm.mean().mul(100).rename("harm_%"),
                d.groupby("format").size().rename("n")], axis=1).round(1))

# ------------------------------------------------------------- the models
rows = []
for name, y in OUTCOMES.items():
    d["_y"] = y.astype(int)
    cl = dict(cov_type="cluster", cov_kwds={"groups": d.paper_id})

    m0 = smf.logit("_y ~ harm", data=d).fit(disp=0, **cl)
    m1 = smf.logit("_y ~ harm + C(evidence_type)", data=d).fit(disp=0, **cl)
    m2 = smf.logit("_y ~ harm + C(evidence_type) + C(format)",
                   data=d).fit(disp=0, **cl)

    # Paper fixed effects: only papers with within-paper variation on BOTH the
    # outcome and the exposure contribute, so report that n.
    grp = d.groupby("paper_id")
    usable = grp.filter(lambda g: g.harm.nunique() > 1 and g._y.nunique() > 1)
    try:
        m3 = ConditionalLogit(usable._y.values,
                              usable[["harm"]].values.astype(float),
                              groups=usable.paper_id.values).fit(disp=0)
        or3, p3 = float(np.exp(m3.params[0])), float(m3.pvalues[0])
    except Exception as e:                      # separation or empty sample
        or3, p3 = np.nan, np.nan

    rows.append(dict(
        outcome=name,
        M0_unadj=round(np.exp(m0.params["harm"]), 2), p0=f"{m0.pvalues['harm']:.1e}",
        M1_evtype=round(np.exp(m1.params["harm"]), 2), p1=f"{m1.pvalues['harm']:.1e}",
        M2_ev_fmt=round(np.exp(m2.params["harm"]), 2), p2=f"{m2.pvalues['harm']:.1e}",
        M3_paperFE=round(or3, 2) if or3 == or3 else None,
        p3=f"{p3:.1e}" if p3 == p3 else "n/a",
        FE_papers=usable.paper_id.nunique(), FE_claims=len(usable),
    ))
show("OR for harmful vs beneficial, under four adjustments",
     pd.DataFrame(rows))

# ------------------------------------------------- stratified + standardised
strat = []
for name, y in OUTCOMES.items():
    d["_y"] = y.astype(int)
    for ev, g in d.groupby("evidence_type"):
        if g.harm.nunique() < 2 or g._y.nunique() < 2:
            continue
        m = smf.logit("_y ~ harm", data=g).fit(disp=0, cov_type="cluster",
                                               cov_kwds={"groups": g.paper_id})
        strat.append(dict(outcome=name, evidence_type=ev, n=len(g),
                          OR=round(np.exp(m.params["harm"]), 2),
                          p=f"{m.pvalues['harm']:.1e}"))
show("stratified by claim specificity (T=taxonomic, D=directional, M=magnitude)",
     pd.DataFrame(strat).pivot(index="outcome", columns="evidence_type",
                               values="OR"))
show("...stratum n", pd.DataFrame(strat).pivot(index="outcome",
                                               columns="evidence_type", values="n"))

# Post-stratification: predicted rate for each valence if the evidence_type mix
# were held at a common distribution (g-computation over the fitted M1).
post = []
for name, y in OUTCOMES.items():
    d["_y"] = y.astype(int)
    m1 = smf.logit("_y ~ harm + C(evidence_type)", data=d).fit(disp=0)
    for target, w in [("observed mix", d.evidence_type.value_counts(normalize=True)),
                      ("equal mix", pd.Series(1/3, index=["D", "M", "T"]))]:
        r = {}
        for h in (0, 1):
            sim = d.copy()
            sim["harm"] = h
            pr = m1.predict(sim)
            r[h] = float(np.average(
                [pr[sim.evidence_type == e].mean() for e in ["D", "M", "T"]],
                weights=[w.get(e, 0) for e in ["D", "M", "T"]]))
        post.append(dict(outcome=name, standardised_to=target,
                         beneficial_pct=round(100*r[0], 1),
                         harmful_pct=round(100*r[1], 1),
                         ratio=round(r[1]/r[0], 2) if r[0] else None))
show("post-stratified rates (g-computation over evidence_type)",
     pd.DataFrame(post))
