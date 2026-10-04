"""Analyses added for the camera-ready revision (reviewer requests).

A1  Table 2 with 95% CIs and contributing sample sizes for every model
A2  The same models restricted to generative-AI claims and to clinical claims
A3  Papers that contribute to the within-paper (fixed-effects) model vs the rest
A4  Added-noise test: how the estimates move when valence labels are randomly
    flipped (non-differential coding error)
A5  The same models by retrieval channel (direct search vs citation chasing)

Data preparation and outcome definitions are identical to 07_pubtype_adjust.py,
so the full-sample estimates reproduce Table 2.

`evidence_type` holds claim specificity, assigned by the extractor under the
priority rule magnitude > directional > taxonomic: M = magnitude (states a
size), D = directional (states a direction), T = taxonomic (states a category
or definition). It is not a study-design variable.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv(BASE / "claims_with_semantics.csv")
enc = pd.read_csv(OUT / "fields_encoded.csv")
df = pd.concat([raw[["claim_id", "paper_id", "year", "work_type", "channel",
                     "evidence_type"]], enc], axis=1)
df["specificity"] = df.evidence_type.map(
    {"T": "taxonomic", "D": "directional", "M": "magnitude"})

wt = df.work_type.fillna("unknown").str.lower()
df["format"] = np.select(
    [wt.str.contains("preprint"), wt.str.contains("conference"),
     wt.str.contains("report"), wt.eq("review"),
     wt.str.contains("article|journal")],
    ["preprint", "conference", "report", "review_crossref", "article"],
    default="unknown")
df["route"] = np.where(df.channel.eq("c1"), "direct search", "citation chasing")
df["genai"] = df.subject_type.eq("generative ai assistant")
df["clinical"] = (df.population_setting.eq("clinical")
                  | df.population_role.eq("clinicians or professionals"))

d = df[df.valence.isin(["beneficial", "harmful"])].copy()
d["harm"] = (d.valence == "harmful").astype(int)
OUTCOMES = {
    "Associative framing": lambda x: x.relation.eq("associates"),
    "Hedged or speculative": lambda x: x.epistemic_strength.isin(["hedged", "speculative"]),
    "Significance-marked": lambda x: x.statistical_marking.eq("significant"),
    "Carries a magnitude": lambda x: x.magnitude_value.ne("missing"),
}


def show(title, obj):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(obj if isinstance(obj, str) else obj.to_string())


def fe_fit(g, ycol="_y", xcol="harm"):
    """Conditional logit on papers with within-paper variation in both x and y."""
    usable = g.groupby("paper_id").filter(
        lambda p: p[xcol].nunique() > 1 and p[ycol].nunique() > 1)
    if usable.paper_id.nunique() < 5:
        return None, usable
    try:
        m = ConditionalLogit(usable[ycol].values, usable[[xcol]].values.astype(float),
                             groups=usable.paper_id.values).fit(disp=0)
        return m, usable
    except Exception:
        return None, usable


def models(g, label):
    """M0-M3 for every outcome on subset g; returns tidy rows."""
    rows = []
    for name, f in OUTCOMES.items():
        g = g.copy()
        g["_y"] = f(g).astype(int)
        cl = dict(cov_type="cluster", cov_kwds={"groups": g.paper_id})
        r = dict(subset=label, outcome=name,
                 claims=len(g), papers=g.paper_id.nunique(),
                 harm_claims=int(g.harm.sum()), outcome_claims=int(g._y.sum()))
        for tag, form in [("M0", "_y ~ harm"),
                          ("M1", "_y ~ harm + C(evidence_type)"),
                          ("M2", "_y ~ harm + C(evidence_type) + C(format)")]:
            try:
                m = smf.logit(form, data=g).fit(disp=0, **cl)
                lo, hi = m.conf_int().loc["harm"]
                r[f"{tag}_OR"] = np.exp(m.params["harm"])
                r[f"{tag}_lo"], r[f"{tag}_hi"] = np.exp(lo), np.exp(hi)
                r[f"{tag}_p"] = m.pvalues["harm"]
            except Exception:
                r[f"{tag}_OR"] = r[f"{tag}_lo"] = r[f"{tag}_hi"] = r[f"{tag}_p"] = np.nan
        m3, usable = fe_fit(g)
        r["M3_papers"], r["M3_claims"] = usable.paper_id.nunique(), len(usable)
        if m3 is not None:
            lo, hi = m3.conf_int()[0]
            r["M3_OR"], r["M3_lo"], r["M3_hi"] = np.exp(m3.params[0]), np.exp(lo), np.exp(hi)
            r["M3_p"] = m3.pvalues[0]
        else:
            r["M3_OR"] = r["M3_lo"] = r["M3_hi"] = r["M3_p"] = np.nan
        rows.append(r)
    return rows


def fmt(r, tag):
    if not np.isfinite(r[f"{tag}_OR"]):
        return "n/e"
    return f"{r[f'{tag}_OR']:.2f} [{r[f'{tag}_lo']:.2f}, {r[f'{tag}_hi']:.2f}]"


def compact(rows):
    return pd.DataFrame([{
        "subset": r["subset"], "outcome": r["outcome"],
        "claims/papers": f"{r['claims']}/{r['papers']}",
        "M0": fmt(r, "M0"), "M1": fmt(r, "M1"), "M2": fmt(r, "M2"),
        "M3": fmt(r, "M3"), "M3 papers/claims": f"{r['M3_papers']}/{r['M3_claims']}",
        "M3 p": f"{r['M3_p']:.1e}" if np.isfinite(r["M3_p"]) else "n/e"} for r in rows])


# ---------------------------------------------------------------- A1
a1 = models(d, "all")
show("A1  Table 2 with 95% CIs (M0-M2 cluster-robust by paper; M3 conditional logit)",
     compact(a1))
pd.DataFrame(a1).to_csv(OUT / "cr_table2_ci.csv", index=False)

# ---------------------------------------------------------------- A2
a2 = []
for label, mask in [("generative AI", d.genai),
                    ("generative AI, 2023+", d.genai & (d.year >= 2023)),
                    ("clinical", d.clinical),
                    ("generative AI x clinical", d.genai & d.clinical)]:
    a2 += models(d[mask], label)
show("A2  subgroup restrictions (n/e = not estimable: fewer than 5 papers with "
     "within-paper variation, or separation)", compact(a2))
pd.DataFrame(a2).to_csv(OUT / "cr_subgroups.csv", index=False)

# ---------------------------------------------------------------- A3
claims_pp = df.groupby("paper_id").size().rename("claims_all")
pp = d.groupby("paper_id").agg(
    year=("year", "first"), route=("route", "first"), fmt=("format", "first"),
    genai=("genai", "any"), harm_share=("harm", "mean"),
    both=("harm", lambda s: s.nunique() == 2)).join(claims_pp)
spec = (df.groupby(["paper_id", "specificity"]).size().unstack(fill_value=0)
          .pipe(lambda t: t.div(t.sum(axis=1), axis=0)))
pp = pp.join(spec)
grp = pp.groupby("both")
a3 = pd.DataFrame({
    "papers": grp.size(),
    "median year": grp.year.median(),
    "% 2025-26": grp.year.apply(lambda s: 100 * (s >= 2025).mean()),
    "% direct search": grp.route.apply(lambda s: 100 * s.eq("direct search").mean()),
    "% preprint": grp.fmt.apply(lambda s: 100 * s.eq("preprint").mean()),
    "% generative AI": grp.genai.mean() * 100,
    "mean claims/paper": grp.claims_all.mean(),
    "% taxonomic claims": grp.taxonomic.mean() * 100,
    "% directional claims": grp.directional.mean() * 100,
    "% magnitude claims": grp.magnitude.mean() * 100,
}).T.round(1)
a3.columns = ["single valence (excluded from M3)", "both valences (contribute to M3)"]
show("A3  papers that can contribute to the within-paper model vs the rest", a3)
a3.to_csv(OUT / "cr_fe_selection.csv")

# ---------------------------------------------------------------- A4
rng = np.random.default_rng(20261002)
a4 = []
for rate in [0.0, 0.05, 0.10, 0.20]:
    for rep in range(1 if rate == 0 else 20):
        n = d.copy()
        if rate:
            flip = rng.random(len(n)) < rate
            n.loc[flip, "harm"] = 1 - n.loc[flip, "harm"]
        for name, f in OUTCOMES.items():
            n["_y"] = f(n).astype(int)
            m0 = smf.logit("_y ~ harm", data=n).fit(disp=0)
            m3, _ = fe_fit(n)
            a4.append(dict(flip_rate=rate, rep=rep, outcome=name,
                           M0=np.exp(m0.params["harm"]),
                           M3=np.exp(m3.params[0]) if m3 is not None else np.nan))
a4 = pd.DataFrame(a4)
a4s = (a4.groupby(["outcome", "flip_rate"])[["M0", "M3"]].median().round(2)
         .unstack("flip_rate"))
show("A4  median OR when a share of valence labels is flipped at random "
     "(20 replicates per rate; 0.0 = observed)", a4s)
a4.to_csv(OUT / "cr_noise_sensitivity.csv", index=False)

# ---------------------------------------------------------------- A5
a5 = []
for label in ["direct search", "citation chasing"]:
    a5 += models(d[d.route.eq(label)], label)
show("A5  by retrieval route", compact(a5))
show("A5  harm share by channel",
     d.groupby("channel").agg(claims=("harm", "size"), papers=("paper_id", "nunique"),
                              harm_pct=("harm", lambda s: round(100 * s.mean(), 1))))
pd.DataFrame(a5).to_csv(OUT / "cr_channel.csv", index=False)

# ---------------------------------------------------------------- A6
# Is harm share inversely related to how often a research area reports tests?
# Unit: outcome type x measurement horizon x population setting cells. The test
# rate is computed among BENEFIT claims only, so that harm claims' own lower
# testing rate (Table 2) does not drive the correlation by construction.
from scipy.stats import spearmanr

d["tested"] = d.statistical_marking.isin(["significant", "not significant"]).astype(int)
KEYS = ["object_type", "measurement_horizon", "population_setting"]


def cell_table(x, min_n=20, min_ben=10):
    g, ben = x.groupby(KEYS), x[x.harm == 0].groupby(KEYS)
    c = pd.DataFrame({"claims": g.size(), "harm_share": g.harm.mean(),
                      "tested_all": g.tested.mean(),
                      "benefit_claims": ben.size(), "benefit_tested": ben.tested.mean()})
    return c[(c.claims >= min_n) & (c.benefit_claims >= min_ben)]


cells = cell_table(d)
rho, p = spearmanr(cells.harm_share, cells.benefit_tested)
rho_all, p_all = spearmanr(cells.harm_share, cells.tested_all)
rng6 = np.random.default_rng(20261003)
rows_of = d.groupby("paper_id").indices
pids = np.array(list(rows_of))
boot = []
for _ in range(1000):
    pick = rng6.choice(pids, len(pids), replace=True)
    idx = np.concatenate([rows_of[q] for q in pick])
    cc = cell_table(d.iloc[idx])
    if len(cc) > 5:
        boot.append(spearmanr(cc.harm_share, cc.benefit_tested)[0])
lo6, hi6 = np.percentile(boot, [2.5, 97.5])
summary6 = pd.DataFrame([{
    "cells": len(cells), "claims_covered": int(cells.claims.sum()), "claims_total": len(d),
    "spearman_rho_benefit_tested": round(rho, 3), "p": round(p, 4),
    "boot_ci_lo": round(lo6, 3), "boot_ci_hi": round(hi6, 3), "boot_reps": len(boot),
    "spearman_rho_all_tested": round(rho_all, 3), "p_all": round(p_all, 4)}])
show("A6  harm share vs share of benefit claims reporting a statistical test, "
     "across outcome x horizon x setting cells (>=20 claims, >=10 benefit claims); "
     "95% CI from 1,000 paper-clustered bootstrap replicates", summary6.T)
for f in ["object_type", "measurement_horizon"]:
    t = d.groupby(f).agg(claims=("harm", "size"), harm_pct=("harm", "mean"))
    t["benefit_tested_pct"] = d[d.harm == 0].groupby(f).tested.mean()
    t = t[t.claims >= 50].mul({"claims": 1, "harm_pct": 100, "benefit_tested_pct": 100}).round(1)
    show(f"A6  by {f} (levels with >=50 claims)", t.sort_values("harm_pct", ascending=False))
cells.round(3).to_csv(OUT / "cr_harm_testing_cells.csv")
summary6.to_csv(OUT / "cr_harm_testing_summary.csv", index=False)
