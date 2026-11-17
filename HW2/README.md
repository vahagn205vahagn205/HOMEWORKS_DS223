# Homework 2: A/B Testing with Multi-Armed Bandits

Four advertisement options (bandits) with mean rewards `[1, 2, 3, 4]` are tested for
20,000 trials with two algorithms:

- **Epsilon-Greedy**: epsilon starts at 1 and decays as `1/t`.
- **Thompson Sampling**: Gaussian rewards with **known precision** (`tau = 1`), Normal prior
  `N(0, 1)` on each bandit's mean, conjugate Normal posterior update.

Each pull returns a reward drawn from `N(mean_of_bandit, 1/precision)`.
Regret per trial is `max(mean) - mean_of_chosen_bandit`.

## How to run

```bash
pip install -r requirements.txt
python Bandit.py
```

All output goes through `loguru` (no `print()`).

## Output

| File | What it contains |
|------|------------------|
| `rewards.csv` | One row per trial, columns `Bandit, Reward, Algorithm` (40,000 rows) |
| `plots/plot1_EpsilonGreedy.png` | Learning process of Epsilon-Greedy (linear and log scale) |
| `plots/plot1_ThompsonSampling.png` | Learning process of Thompson Sampling (linear and log scale) |
| `plots/plot2_cumulative_reward_regret.png` | Cumulative reward and cumulative regret of both algorithms |
| `plots/comparison.png` | Pulls per bandit and average regret over time for both algorithms |

The cumulative reward, cumulative regret, average reward, average regret and pulls per
bandit are logged for each algorithm, followed by a summary table.

## Results (seed = 42)

| Algorithm | Cumulative reward | Cumulative regret | Best bandit pulled |
|-----------|------------------:|------------------:|-------------------:|
| Epsilon-Greedy | 73,907.23 | 6,181.00 | 69.13% |
| Thompson Sampling | 79,967.44 | 7.00 | 99.98% |

Thompson Sampling finds bandit 4 almost right away. Epsilon-Greedy gets stuck on bandit 3
for about 6,000 trials. The reason is that `1/t` makes epsilon very small very fast
(after 100 trials it is already 0.01), so there is almost no exploration left. Bandit 4
was never tried early, so its estimate stayed at 0 and greedy choice kept picking bandit 3
until a random exploration step finally hit bandit 4.

## Code structure

- `Bandit`: the abstract base class from the template (nothing removed).
- `EpsilonGreedy(Bandit)` and `ThompsonSampling(Bandit)`: each object runs one algorithm
  over all four bandits. Methods: `pull`, `update`, `experiment`, `report`.
- `Visualization`: `plot1()` (learning process) and `plot2()` (cumulative reward and regret).
- `comparison()`: extra visual and table comparison of the two algorithms.

Docstrings use the reStructuredText style that `pyment` produces.

## Bonus: a better implementation plan

1. **Average over many runs.** One run with one seed can be lucky or unlucky (like
   Epsilon-Greedy above). Running each algorithm, for example, 100 times with different
   seeds and plotting the mean regret with a confidence band gives a fair comparison.
2. **Fix the exploration problem of Epsilon-Greedy.**
   - Pull every bandit once at the start, so no estimate stays at its starting value.
   - Use a slower decay such as `min(1, c * K / t)` with a tuned `c`, or keep a small
     minimum epsilon.
   - Or use optimistic starting estimates (for example 10), which makes the algorithm try
     every bandit by itself.
3. **Add UCB1 as a third baseline.** It explores using confidence bounds instead of random
   choice and has a known `O(log T)` regret bound.
4. **Do not assume the precision is known.** In real A/B tests the noise level is unknown.
   A Normal-Gamma prior lets Thompson Sampling learn the mean and the precision together.
   If the reward is a click (0 or 1), a Beta-Bernoulli model is the natural choice.
5. **Cleaner code.**
   - Move the shared loop (`experiment`) and shared `report` into one middle class, so each
     algorithm only defines how it picks a bandit and how it updates.
   - Put the settings (means, trials, epsilon, precision, seed) in a config dataclass or
     command-line arguments instead of constants.
   - Add `pytest` tests: posterior update formula, epsilon schedule, CSV columns, and that
     each algorithm ends up picking the best bandit most of the time.
6. **Compare with a classic A/B test.** Splitting traffic evenly (25% each) for a fixed time
   and then picking the winner shows how much regret the bandit methods save.
