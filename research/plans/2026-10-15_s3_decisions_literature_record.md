# S3 waiting decisions: literature record (2026-10-15)

Question set (S1 record section 4): (1) number of permutations and cross-fit seeds to register; (2) Westfall-Young or Benjamini-Hochberg for secondary statistics. This record separates what was read in full text, what was only seen as a snippet, and what is the author's own derivation. It decides nothing; the researcher decides.

**Method.** Three independent literature scouts (neutral questions, no repository access, web only) returned source maps at snippet or abstract depth. The author then downloaded eight open-access PDFs into the library (`C:/Users/ASUS/Desktop/QML Research/papers/Stats/`, index file there) with the researcher's permission and read the implementation-relevant passages. Two sources could not be downloaded (links in section 5).

## 1. Verified in full text (locators are in the extracted text; page numbers to be added when read in the PDF viewer)

| claim | source | status |
|---|---|---|
| With B random permutations and the observed statistic counted as one more, the exact Monte Carlo p-value is (b+1)/(B+1). The naive b/B gives a test whose type I error exceeds the nominal rate for most practical values and is never below 1/(m+1) whatever the rate chosen; published p-values are "understated by about 1/m". It also gives exact zeros that break multiplicity procedures. | Phipson and Smyth 2010 (arXiv 1603.05766), sections 3-4 (type I error of p-hat; "pu = P(B <= b) = (b+1)/(m+1)") | read |
| Drawing permutations with replacement makes the exact p-value slightly smaller than (b+1)/(m+1); (b+1)/(m+1) is "valid but conservative" and over-estimates a 0.05 p-value by about 2% at 1000 distinct values, more for small p, very conservative for few distinct values | Phipson and Smyth 2010, section 6.2 | read |
| The 2010 proof is "incomplete" because it does not use the group structure of the permutations; the result (b+1)/(B+1) is stochastically larger than uniform under the null when the transformations form a group. Random sampling from balanced permutations (not a group) can give an invalid test. Naive b/B with Bonferroni "can lead to completely faulty inference". | Hemerik and Goeman 2018 (arXiv 1411.7565), introduction and sections 3.2-3.3 | read. This resolves the scout's flagged "disagreement": it concerns the proof, not the estimator |
| Monte Carlo standard deviation of a permutation p-value is sqrt(p(1-p)/k) for k samples | Ojala and Garriga 2010 (JMLR 11), section 2, quoting Efron-Tibshirani 1993 and Good 2000 | read. Their experiments used 1000 randomisations; that is an experimental choice, **not a recommendation** (no recommended count found in the text) |
| An open-ended sequential Monte Carlo test with uniformly bounded resampling risk (probability of a decision different from the one based on the exact p-value is bounded by an arbitrary constant); earlier bounded-sample sequential and non-sequential algorithms do not have this property | Gandy 2009 (arXiv math/0612488), abstract | abstract read; algorithm details not studied |
| No universal unbiased estimator of the variance of K-fold cross-validation exists (Theorem 6); naive estimators ignoring the overlap correlations "grossly underestimate variance" | Bengio and Grandvalet 2004 (JMLR 5), abstract and Theorem 6 | statement read; proof not studied |
| Standard cross-validation confidence intervals can have coverage far below nominal because fold accuracies are correlated; a nested cross-validation scheme estimates the variance better with approximately correct coverage in many (not all) of their examples; cross-validation estimates the average error over training sets, not the error of the fitted model | Bates, Hastie and Tibshirani 2023 (arXiv 2104.00673), abstract and introduction | abstract and introduction read; theorems and simulation settings not studied |
| BH controls the FDR under positive regression dependence (PRDS) (Proposition 1, attributed there to Benjamini-Yekutieli 2001); for weakly negatively dependent null p-values BH can be anti-conservative, with FDR at most about alpha times min(-ln alpha + 3.18, K) (Theorem 19, asymptotically tight) | Chi, Ramdas and Wang 2024 (arXiv 2212.09706), section 5.1 | read; the symbols in the extracted text are garbled, so the exact form of the cap must be confirmed in the PDF before use |

## 2. Seen only at snippet or abstract depth, or secondhand (not usable as evidence yet)

- Westfall-Young step-down maxT/minP and the subset-pivotality condition; the dispute (Westfall-Troendle 2008 "hardly restrictive" against Romano-Shaikh-Wolf 2007 "quite restrictive") is known only from secondary snippets. Romano-Wolf 2005 stepdown without subset pivotality: snippet only.
- Holm (1979): strong FWER control under any dependence is textbook knowledge, primary not read.
- Benjamini-Yekutieli 2001 theorem statements: seen only through Chi et al.'s citation.
- Nadeau-Bengio 2003 corrected resampled t-test; Dietterich 1998 (5x2cv); Bouckaert-Frank 2004; Varoquaux 2018; Varma-Simon 2006; Cawley-Talbot 2010; Kim 2009: abstract or secondary only.
- Cai et al. 2307.00260 (bootstrap estimate of the cross-validation standard error): downloaded; only the abstract was read. The scout's summary that repeats reduce only the fold-assignment component of the variance is **unverified**.
- Gross-Vitells 2010 (trial factors in particle physics), Demsar 2006, LeDell-Petersen-van der Laan 2015 (cross-validated AUC intervals), DeLong 1988, Brown-Cai-DasGupta 2001: not read.
- The scouts did not find: a sourced recommended number of permutations for classifier-performance tests; a sourced number of cross-validation repeats ("10 or more" is unsupported); a power comparison of Holm, BH and Westfall-Young for correlated statistics; any treatment of repeated cross-validation at very small false-positive fractions. These are **gaps in what was found**, not evidence of absence (the scopes were web snippets and a short list of named papers).

## 3. What follows for the two decisions (options and reasoning, not decisions)

**Permutation count B.** Established: use (b+1)/(B+1), never b/B; stratified within-stratum permutation is a group (a product of symmetric groups; the author's inference, to be confirmed by the statistics reviewer), so the Hemerik-Goeman validity condition is plausible but must be argued in the registration. Author's derivations (not from a source): the smallest attainable p-value is 1/(B+1); Holm across m tests at family-wise 0.05 needs 1/(B+1) <= 0.05/m, i.e. B >= 20m - 1 (m = 192, the unadjusted count of the earlier audit, gives B >= 3,839, so the code default of 1,000 is too small for Holm at that m); the Monte Carlo standard error at p = 0.05 is 0.005 for B of about 1,900 and 0.0022 for B = 10,000. The trade-off that only the researcher can weigh is the cost per permutation of refitting the classifier. If cost is high, Gandy's sequential rule gives a decision with bounded resampling risk instead of a fixed B (abstract-level evidence only).

**Cross-fit seeds.** Established: the spread over seeds is not the uncertainty of the AUC; the naive variance is too small (Bengio-Grandvalet Theorem 6; Bates et al. 2023), so a variance estimator that accounts for correlation (nested cross-validation, or a bootstrap of the whole procedure) is needed in addition to seed repeats. Not established: any specific number of repeats; the code default of 5 has no literature support found, and the scouts' "10 or more" has none.

**Westfall-Young or BH for secondaries.** Established: BH has a proof under PRDS; under negative dependence it can exceed its nominal FDR by a bounded factor (Theorem 19); the features of this project are mostly positively correlated but sign structure is unchecked (a check is feasible on the exposed data: sign of the pairwise correlations of the null permutation statistics). Westfall-Young needs all permutation statistics in one pass (S1 record) and its validity condition is disputed in sources not yet read. Holm needs no dependence assumption but is conservative. The choice between FWER (Holm or Westfall-Young) and FDR (BH) is a statement about what a false secondary finding costs in the paper; no source found addresses that for exploratory secondaries.

## 4. Open items before the decision is fully sourced

1. Read Benjamini-Yekutieli 2001 (PRDS theorem) and Westfall-Troendle 2008 (links below), then Romano-Wolf 2005 (paywalled).
2. Read the Cai et al. bootstrap paper beyond the abstract, and the Bates et al. nested scheme, if the cross-fit uncertainty route is chosen.
3. Check the sign structure of the correlations among the secondary statistics on the exposed data (cheap, no sealed data).
4. Independent statistics review of the registered permutation design (SV-9), when the researcher releases the reviewers.

## 5. Not downloaded (links for the researcher)

- Benjamini and Yekutieli 2001, Annals of Statistics 29(4):1165-1188, DOI 10.1214/aos/1013699998 (Project Euclid returned a web page, not the PDF).
- Westfall and Troendle 2008, Biometrical Journal 50(5):745-755, PMC3117234 (PMC returned a web page).
- Paywalled, not attempted: Romano and Wolf 2005 (JASA 100:94-108), Nadeau and Bengio 2003 (Machine Learning 52:239-281), the Westfall-Young 1993 book, Besag-Clifford 1991.

## 6. Sign-structure check (run 2026-10-15; exploratory diagnostic, exposed data only)

Script `research/reviews/2026-10-15_s3_sign_structure.py`, output `research/reviews/2026-10-15_s3_sign_structure.json`. Scope fixed before the run: electrons and protons of the development-large sets at 20 and 100 GeV (1,500 events per class, no sealed data), 32 features (the conventional features plus the multiscale table), 500 unstratified label permutations, seed 20261015. Statistic examined: the per-feature KS distance between classes, as used in the earlier family-wise check.

| quantity | 20 GeV | 100 GeV |
|---|---|---|
| features whose own pairwise correlations are negative (Spearman, pooled events; 496 pairs) | 51% of pairs; 32% below -0.2; minimum -0.93 | 49%; 32% below -0.2; minimum -0.94 |
| null KS statistics (500 permutations) with a negative pairwise correlation | 11% of pairs | 9% |
| null KS pairs below -2 standard errors (single-pair SE 0.045) | 0.8% (4 of 496) | 0% |
| null KS pairs above +2 standard errors | 56% | 64% |
| smallest null KS correlation | -0.12 | -0.08 |

**Reading.** The features themselves are strongly correlated with mixed signs, but the KS distances (unsigned, non-negative statistics) are almost all positively associated under the permutation null; the few negative pairs sit inside Monte Carlo noise (a pair at -0.12 is 2.7 standard errors, and one such pair is expected among 496 by chance). So for this set of statistics the negative-dependence regime of Chi et al. does not appear, which favours the cheap positive-dependence analysis. **Limits:** (1) a non-negative linear correlation of the statistics is **not** a proof of positive regression dependence (PRDS), which concerns the joint behaviour of the p-values; (2) only unstratified permutations, two energies and the electron-versus-proton KS statistics were examined, not the AUC-gain, ablation or variant-comparison secondaries of S3; (3) the multiscale features with many ineligible events were kept when at least half the events were finite, which is a choice made here; (4) 500 permutations resolve correlations only down to about plus or minus 0.09. The check informs the choice between BH and Westfall-Young; it does not make it.
