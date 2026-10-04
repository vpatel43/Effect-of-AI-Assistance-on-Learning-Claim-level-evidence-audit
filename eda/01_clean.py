"""Step 0: normalise the 19 semantic columns and derive the analysis frame.

Reads claims_with_semantics.csv, writes claims_clean.csv (+ a stated/not-stated
mask). Everything downstream reads claims_clean.csv, never the raw file.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"

# The 11 closed-vocabulary facets. subject/object_entity, mediator, moderator,
# magnitude_* and n_reported are open text or numeric and handled separately.
FACETS = [
    "subject_type", "subject_mode", "object_type", "relation", "direction",
    "valence", "population_role", "population_setting", "measurement_horizon",
    "statistical_marking", "epistemic_strength",
]

# Label variants observed in the raw file that fragment the crosstabs.
FIXES = {
    "subject_type": {"non_AI_external_aid": "non-AI external aid"},
    "subject_mode": {"replacement or removal": "removal or withdrawal"},
    "object_type": {
        "behavioural_strategy": "behavioural strategy",
        "behavioral strategy": "behavioural strategy",
        "task_performance": "task performance",
        "mixed": "mixed or multiple",
    },
}

# Values that mean "the abstract did not say", as opposed to a real category.
UNSTATED = {"not_stated", "not_applicable"}


def load():
    df = pd.read_csv(BASE / "claims_with_semantics.csv")

    for col, mapping in FIXES.items():
        df[col] = df[col].replace(mapping)

    # measurement_horizon has a single genuine NaN; treat as not_stated.
    df[FACETS] = df[FACETS].fillna("not_stated")

    # Open-text columns: strip, lowercase-fold for matching, keep original.
    for col in ["subject_entity", "object_entity", "mediator", "moderator"]:
        df[col] = df[col].fillna("not_stated").astype(str).str.strip()
        df[col + "_norm"] = df[col].str.lower()

    return df


def derive(df):
    # Stated/not-stated mask over the closed facets.
    mask = pd.DataFrame(
        {c: ~df[c].isin(UNSTATED) for c in FACETS}, index=df.index
    )
    df["n_stated"] = mask.sum(axis=1)
    df["n_unstated"] = len(FACETS) - df["n_stated"]

    # Separate "not_applicable" (a coding decision) from "not_stated" (silence),
    # so missingness models can distinguish them.
    df["n_not_stated"] = (df[FACETS] == "not_stated").sum(axis=1)
    df["n_not_applicable"] = (df[FACETS] == "not_applicable").sum(axis=1)

    # Compact claim signature: the causal shape of the assertion.
    df["signature"] = (
        df.subject_type + " | " + df.subject_mode + " -> "
        + df.relation + "/" + df.direction + " -> " + df.object_type
    )
    # Coarser: just what acts on what, in which direction.
    df["core_shape"] = df.relation + "/" + df.direction + "/" + df.valence

    # Does the claim carry an explicit mechanism slot?
    df["has_mediator"] = ~df.mediator.isin(UNSTATED)
    df["has_moderator"] = ~df.moderator.isin(UNSTATED)

    # Quantitative payload actually usable?
    df["has_magnitude"] = df.magnitude_value.notna()

    # work_type is a mix of Crossref and PubMed vocabularies; collapse.
    wt = df.work_type.fillna("unknown").str.lower()
    df["work_type_c"] = np.select(
        [
            wt.str.contains("meta-analysis|systematic review", regex=True),
            wt.str.contains("review"),
            wt.str.contains("randomized|randomised"),
            wt.str.contains("preprint"),
            wt.str.contains("conference"),
            wt.str.contains("article|journal"),
        ],
        ["synthesis", "review", "rct", "preprint", "conference", "article"],
        default="other",
    )
    return df, mask


if __name__ == "__main__":
    df = load()
    df, mask = derive(df)
    df.to_csv(OUT / "claims_clean.csv", index=False)
    mask.to_csv(OUT / "stated_mask.csv", index=False)

    print(f"rows {len(df)}  papers {df.paper_id.nunique()}")
    print(f"\nfacets stated per claim (of {len(FACETS)}):")
    print(df.n_stated.value_counts().sort_index().to_string())
    print("\nstated rate by facet:")
    print((mask.mean().sort_values(ascending=False) * 100).round(1).to_string())
    print("\nwork_type_c:")
    print(df.work_type_c.value_counts().to_string())
