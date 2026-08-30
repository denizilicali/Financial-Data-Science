# Write-up markdown file — PS 5

## Task 1 — Data Inspection

The dataset has 17,402 rows and 198 columns (excluding the target). 184 of these are news
dummies and 14 are controls (const, sigma_d, day dummies, hour dummies, window dummies).

The target y_diff_squared has a mean of 1.38e-08, std of 2.33e-07, min of 0 and max of
1.94e-05. The std is much larger than the mean and the max is very far from the average,
so most bars are close to zero and only a few bars have large values. This makes sense
since markets are usually calm and only move a lot around specific events.

---

## Task 9 — Write-up

### 1. LASSO Diagnostics

The LASSO selected 14 news variables out of 184. The CV-picked penalty was
alpha_ = 7.15e-09 and alphas_[0] (the largest on the grid) was 3.82e-08. Since
alpha_ < alphas_[0], CV did not pick the trivial solution and actually selected
variables. The shuffled KFold was important here — without it news events would cluster
in time and CV would pick alpha_max, selecting nothing.

### 2. Top-3 News Dummies

1. **Central Bank_BoE_Minutes** — t-stat = +3.32. The Bank of England Minutes release
   moves markets because it reveals how the MPC members voted and what they discussed,
   giving new information about future rate decisions.

2. **Ad Hoc_Covid-19** — t-stat = +2.31. Covid-19 related news caused large spikes in
   variability, which makes sense given how unexpected and severe the pandemic was in
   early 2020.

3. **Central Bank_ECB_Ad Hoc Press Release** — t-stat = +1.37. Unscheduled ECB press
   releases usually mean something unusual is happening, so markets react more strongly
   than to normal scheduled releases.

### 3. Category Ranking

| Category      | omega_GKW |
|---------------|-----------|
| Central Bank  | 0.1772    |
| Ad Hoc        | 0.1266    |
| Macro Release | 0.0760    |
| Auction       | 0.0000    |
| **Total**     | **0.3798** |

The four values sum to 0.3798, which equals the overall omega_GKW, so the
winner-takes-all implementation is correct.

### 4. Judgement

Central Bank having the highest share is not surprising since central bank communication
is known to be one of the biggest drivers of financial markets, but I did not expect
Ad Hoc to rank second above Macro Releases.
