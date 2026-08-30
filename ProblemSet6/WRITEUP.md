# PS 6 -- Write-up

**1. OOS R^2 ranking.** Best to worst, all seven models: PCR (0.0273) >
PLS (0.0256) > ElasticNet (0.0223) > Huber (0.0208) > OLS (0.0176) >
RandomForest (0.0029) > BoostedTrees (-0.0573). So PCR and PLS do the
best, the linear/regularized models (ElasticNet, Huber, OLS) are right
behind, and the two tree models are at the bottom, with BoostedTrees even
going negative. This makes sense because the true model here is just
linear in the 15 features, with a pretty weak signal (only ~5% R^2).
Linear models and PCR/PLS are basically built for this kind of problem, so
they do fine. Trees are made to catch non-linear patterns and
interactions, but there aren't any here, so they mostly just fit noise.
RandomForest still gets a tiny positive R^2 since averaging many trees
calms it down, but BoostedTrees clearly overfits the small training set
(~1,500 rows) and ends up worse than just guessing the mean.

**2. Diebold-Mariano significance.** A bunch of |DM| values are above
1.96. Some examples: (a) PCR vs. BoostedTrees, DM = -4.30, so PCR is
significantly better than BoostedTrees. (b) OLS vs. Huber, DM = 2.43, so
Huber is significantly better than OLS (probably because Huber is more
robust to weird/outlier returns). (c) RandomForest vs. PCR, DM = 2.82, so
PCR beats RandomForest by a significant margin. (d) RandomForest vs.
BoostedTrees, DM = -3.74, so RandomForest is significantly better than
BoostedTrees, even though RandomForest itself isn't a great model overall.

**3. Model selection.** ElasticNet picked alpha=0.01 and l1_ratio=0.1,
which is a really small, mostly ridge-like penalty -- basically not much
different from plain OLS. PCR and PLS both picked just 1 component out of
15. That's a bit odd since the real model actually uses all 15 features.
I think what's happening is that with a weak signal and limited data, the
validation set can't really tell the difference between the extra
components' gains and just noise, so it plays it safe and picks the
simplest option. So the models don't find the "true" complexity, but they
make a reasonable choice given how hard this prediction problem is.

**4. Practical judgement.** OOS R^2 treats every stock the same and mostly
just reflects how close you get on the "average" prediction, but for a
long-short portfolio what really matters is whether you correctly rank the
best and worst stocks each month, not the overall prediction error, so a
model could look bad in R^2 terms but still be totally fine (or even good)
for building the actual long-short trades.
