"""Feasibility probe for the mechanism graph.

subject_entity has 3354 distinct strings, object_entity 4588. A mechanism graph
only works if those collapse to a few hundred concepts AND if one claim's object
matches another claim's subject often enough to form chains. This measures both
before committing to the full normalisation pass.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "eda"
pd.set_option("display.width", 200)

df = pd.read_csv(OUT / "claims_clean.csv")
UNSTATED = {"not_stated", "not_applicable"}


def show(title, obj):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(obj if isinstance(obj, str) else obj.to_string())


# ------------------------------------------------------- how long are these?
for col in ["subject_entity", "object_entity", "mediator", "moderator"]:
    s = df.loc[~df[col].isin(UNSTATED), col]
    wl = s.str.split().str.len()
    show(f"{col}: length profile",
         f"n={len(s)}  distinct={s.nunique()}\n"
         f"words  median={wl.median():.0f}  p90={wl.quantile(.9):.0f}  "
         f"max={wl.max():.0f}\n"
         f"share >8 words (prose, not a term): {(wl > 8).mean()*100:.1f}%")

# --------------------------------------------- raw string overlap, no NLP yet
subj = df.loc[~df.subject_entity_norm.isin(UNSTATED), "subject_entity_norm"]
obj = df.loc[~df.object_entity_norm.isin(UNSTATED), "object_entity_norm"]
show("exact-string chaining potential (no normalisation)",
     f"distinct subjects: {subj.nunique()}   distinct objects: {obj.nunique()}\n"
     f"strings appearing as BOTH a subject and an object: "
     f"{len(set(subj) & set(obj))}\n"
     f"claims whose object string is some other claim's subject string: "
     f"{obj.isin(set(subj)).sum()} of {len(obj)}")

# ----------------------------------- how concentrated is the entity vocabulary?
for name, s in [("subject_entity", subj), ("object_entity", obj)]:
    vc = s.value_counts()
    cum = vc.cumsum() / vc.sum()
    show(f"{name}: vocabulary concentration",
         f"top 50 strings cover {cum.iloc[49]*100:.1f}% of mentions\n"
         f"top 200 strings cover {cum.iloc[199]*100:.1f}% of mentions\n"
         f"singletons: {(vc == 1).sum()} ({(vc==1).mean()*100:.1f}% of vocab, "
         f"{vc[vc==1].sum()/vc.sum()*100:.1f}% of mentions)")

# ------------------------------------- can TF-IDF + agglomerative collapse it?
# Character n-grams catch morphological variants (ChatGPT / chatgpt / GPT-4);
# word features catch synonym-ish phrasing. This is a lower bound on what an
# embedding model would achieve.
def probe_cluster(series, n_clusters):
    vc = series.value_counts()
    terms = vc.index.tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = normalize(vec.fit_transform(terms))
    # Cosine distance from the sparse matrix; densifying X itself would be ~GBs.
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    lab = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                  linkage="average").fit_predict(D)
    out = pd.DataFrame({"term": terms, "n": vc.values, "cluster": lab})
    g = out.groupby("cluster").agg(
        mentions=("n", "sum"), n_terms=("term", "size"),
        exemplar=("term", "first"))
    return out, g.sort_values("mentions", ascending=False)


sub_terms, sub_g = probe_cluster(subj, 120)
show("subject_entity collapsed to 120 clusters — top 25 by mentions",
     sub_g.head(25))
show("subject cluster size profile",
     f"clusters holding >=10 mentions: {(sub_g.mentions>=10).sum()} of 120\n"
     f"largest cluster: {sub_g.mentions.iloc[0]} mentions, "
     f"{sub_g.n_terms.iloc[0]} distinct strings")

obj_terms, obj_g = probe_cluster(obj, 150)
show("object_entity collapsed to 150 clusters — top 25 by mentions",
     obj_g.head(25))

# Inspect a couple of clusters to judge whether the merges are sane.
for cid in sub_g.index[:4]:
    members = sub_terms[sub_terms.cluster == cid].nlargest(12, "n")
    show(f"subject cluster {cid} members (top 12)", members[["term", "n"]])

sub_terms.to_csv(OUT / "subject_entity_clusters.csv", index=False)
obj_terms.to_csv(OUT / "object_entity_clusters.csv", index=False)
