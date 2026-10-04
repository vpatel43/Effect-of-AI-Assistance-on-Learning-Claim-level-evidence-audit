# Under-reported Harm: claim-level evidence audit

Data and analysis code for:

> Patel VN. **Under-reported Harm: A Claim-level Audit of Abstract Reporting on
> AI Assistance and Human Learning.** Human-AI Coevolution (HAIC) workshop,
> NeurIPS 2026.

This repository contains the claim-level corpus analysed in the paper and the
analysis scripts that produce every number in the Results, Discussion and
Appendix.

---

## Contents

```
claims_with_semantics.csv           6,063 coded claims x 39 columns  (the analytic input)
extracted_claims_all_included.csv   6,075 claims before schema coding (16 columns)
evidence_to_claims_reproduction.ipynb   record of corpus construction (see caveat below)
supplemental_methods/               artifacts behind the paper's Appendix A
eda/01_clean.py ... 09_*.py         the analysis pipeline, run in numeric order
eda/10_camera_ready.py              analyses added in revision (Table 2 CIs, Appendix C, harm-testing correlation)
eda/*.txt                           saved console output from each script
eda/*.csv, eda/*.npy                derived tables and coordinates
requirements.txt                    Python dependencies
LICENSE                             MIT
```

`claims_with_semantics.csv` carries the 19 schema fields described in Table 1 of
the paper, plus bibliographic and provenance columns (`paper_id`, `doi`, `year`,
`venue`, `work_type`, `channel`, `retrieval_depth`, `extractor`,
`source_retracted`). One row per claim; cluster on `paper_id` for any
inferential model, as the paper does.

### A note on `evidence_type`

The `evidence_type` column records **claim specificity**, assigned by the
extractor under a priority rule (magnitude before directional before taxonomic):

| Code | Meaning |
|---|---|
| `T` | taxonomic: a category or definition is coherent and usable |
| `D` | directional: an effect exists and runs in a stated direction |
| `M` | magnitude: an effect has a stated size, or holds under a stated moderator |

It describes the claim, not the study that produced it, and the corpus contains
no reliable study-design variable. The submitted version of the paper and of
these scripts read the same codes as study design (direct-empirical / synthesis
/ theoretical); the camera-ready corrects that reading in the text and in the
script labels. No estimate changes, because the coded values are the same.

## Reproducing the results

```
python -m venv venv && ./venv/bin/pip install -r requirements.txt
cd eda
for s in 0*.py 10_camera_ready.py; do ../venv/bin/python "$s"; done
```

The scripts must run in numeric order: later ones read intermediates written by
earlier ones (`07` and `10` need `fields_encoded.csv` from `05`). They read the
corpus by relative path (`../claims_with_semantics.csv`), so keep the directory
layout as shipped. Total runtime is under two minutes on a laptop CPU, dominated
by `06_claim_structure.py` (~70 s; it fits latent class models for K = 2..20 with
three restarts each). No GPU and no network access are required.

`statsmodels` emits `ConvergenceWarning` for a few of the sparse-cell logistic
models. This is present in the saved logs as well and does not affect the
estimates reported in the paper, which reproduce exactly.

## Which file produces which result

| Paper artifact | Script | Output to check |
|---|---|---|
| Table 2, odds ratios with 95% CIs and contributing papers / claims | `10_camera_ready.py` (point estimates also in `07`) | `cr_table2_ci.csv`, `10_camera_ready.txt` |
| Table 3, raw vs specificity-standardised distributions | `08_balanced_and_gaps.py` | `08_balanced_and_gaps.txt` |
| Table 4, under-represented cells in the intervention x outcome x setting cube | `08_balanced_and_gaps.py` | `gap_cells.csv`, `gap_cube_horizon.csv` |
| Table A1, field cardinality and empty rates | `02_coverage.py`, `03_missingness.py` | `02_coverage.txt` |
| Table A2, latent class profiles (K = 11) | `06_claim_structure.py` | `claims_lca.csv`, `lca_selection.csv` |
| Table A3, generative-AI and clinical subsets | `10_camera_ready.py` | `cr_subgroups.csv` |
| Table A4, papers contributing to the within-paper model vs the rest | `10_camera_ready.py` | `cr_fe_selection.csv` |
| Table A5, added-noise test on valence labels | `10_camera_ready.py` | `cr_noise_sensitivity.csv` |
| Table A6, estimates by retrieval route | `10_camera_ready.py` | `cr_channel.csv` |
| Abstract and Sec. 3.3, harm share vs quantitative reporting (Spearman rho -0.46, 95% CI -0.59 to -0.11, across 25 outcome x horizon x setting cells) | `10_camera_ready.py` (section A6) | `cr_harm_testing_summary.csv`, `cr_harm_testing_cells.csv` |
| Appendix B.2, MCA dimensions and contributions | `06_claim_structure.py` | `mca_field_contributions.csv`, `mca_claim_coords.npy` |
| Appendix B.4, adjusted mutual information | `05_field_structure.py` | `field_ami.csv`, `field_ami_costated.csv` |
| Sec. 3.1, extractor invariance (TVD 0.06 / 0.11) | `03_missingness.py` | `instrument_tvd.csv` |
| Sec. 3.3, 82.1% vs 70.8% reported without a test; 39.8% and 37.4% harm shares; margin-balanced harm shares | `09_why_benefit_surplus.py` | `09_why_benefit_surplus.txt` |
| Sec. 3.5, joint coverage (34.7%, 8.7%) | `08_balanced_and_gaps.py` | `08_balanced_and_gaps.txt` |
| Sec. 4, harm decomposition (496 / 190 claims) | `05_field_structure.py` | `05_field_structure.txt` |

Figure 1 is drawn in the paper source from the counts reported in Section 2.1;
it is not generated by these scripts.

## Supplemental methods (paper Appendix A)

| Paper artifact | File |
|---|---|
| Appendix A.2, per-field reliability gate: kappa, source-blocked 95% bootstrap interval, population rate and keep/drop decision for all 11 categorical fields | `supplemental_methods/A2_reliability_gate.csv` |
| Section 2.5 and Appendix A.2, validation of the claim-specificity rule (kappa -0.207 under unordered glosses, 0.634 under the ordered rule, 0.667 with a shared codebook) | `supplemental_methods/A2_extractor_validation.json` |

Both files were extracted from the saved cell outputs of
`evidence_to_claims_reproduction.ipynb` and every value that also appears in the
paper was checked against it. The reliability table includes the two fields the
paper mentions without giving intervals: `measurement_horizon`
(kappa 0.405 [0.168, 0.596], dropped as too sparse) and `epistemic_strength`
(kappa 0.683 [0.447, 0.838]).

Appendix A.1, the pre-window exception rule, has no data artifact in this
repository. The counts it reports -- 2,075 deferred records, condition (i) decidable
for 2,055 of them, reference lists backfilled for 152 included records, and 704
records meeting condition (i) -- were produced during corpus construction and
depend on the citation-linkage step described below, which is not distributed.
The rule itself, including the preregistered threshold of two citing in-window
sources, is stated in full in Appendix A.1 of the paper.

## What this repository does not reproduce

The notebook documents corpus construction — search, screening, claim
extraction and schema coding — and retains its saved cell outputs, including the
integrity assertions described in the Methods. It is **not executable from this
repository**. It imports an internal package that is not distributed here and reads
cached API responses and intermediate JSONL artifacts that are not included,
because they contain full retrieved records rather than the coded claims. It is
provided as a record of how `claims_with_semantics.csv` was produced, not as a
runnable pipeline.

Reproducing corpus construction end to end would additionally require live API
access to OpenAlex, PubMed, arXiv, Semantic Scholar and Crossref, plus the two
language-model APIs named in the Methods. Every result in the paper is
reproducible from `claims_with_semantics.csv` using `eda/` alone.

## Provenance and limitations

All claims derive from abstracts; the corpus contains no full-text retrievals.
The sentinel values `not_stated` and `not_applicable` record what an abstract
omitted, not what a study failed to measure. In particular, a harm claim with
`statistical_marking = not_stated` may come from a study that tested the outcome
but did not report the test in its abstract. See the Limitations section of the
paper before drawing inferences from empty fields.

Schema coding was performed by a language model, with cross-family inter-model
agreement reported per field in Appendix A.2. These are agreement figures, not
accuracy against expert judgement; a blinded human-coded validation sample is
planned.

Console logs in `eda/*.txt` had absolute filesystem paths replaced with `<env>`
and `<path>`. No other content was altered; the numeric output is unchanged.

## Citing

If you use the corpus or the scripts, please cite the paper above.

## License

MIT. See `LICENSE`.
