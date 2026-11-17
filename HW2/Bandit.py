"""
A/B testing of four advertisement options with two multi-armed bandit algorithms:
Epsilon-Greedy (epsilon decays as 1/t) and Thompson Sampling (Gaussian rewards
with known precision).

Every log message goes through loguru instead of print().

Running the file will:
    1. run both experiments for 20000 trials,
    2. save the trial-level rewards to ``rewards.csv`` with the columns
       Bandit, Reward, Algorithm,
    3. log the cumulative reward and cumulative regret of each algorithm,
    4. save the plots to the ``plots/`` folder (and show them when a display is available).
"""
############################### LOGGER
import os
import sys
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
EPSILON = 1.0            # starting epsilon, decayed as EPSILON / t
PRECISION = 1.0          # known precision (1 / variance) of the reward noise
SEED = 42
CSV_PATH = "rewards.csv"
PLOT_DIR = "plots"


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plots for one or more finished bandit experiments.

    :param algorithms: algorithm objects that have already run ``experiment()``
    :type algorithms: list
    :param plot_dir: folder where the figures are saved
    :type plot_dir: str
    """

    def __init__(self, algorithms, plot_dir=PLOT_DIR):
        """Constructor method."""
        self.algorithms = algorithms
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def _finish(self, fig, file_name):
        """Save a figure, show it if the backend is interactive, then close it.

        :param fig: figure to save
        :type fig: matplotlib.figure.Figure
        :param file_name: name of the image file inside ``plot_dir``
        :type file_name: str
        """
        path = os.path.join(self.plot_dir, file_name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        logger.info(f"Saved plot: {path}")
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        """Learning process of each algorithm.

        One figure per algorithm. The top row shows the running average
        reward on a linear and on a log x-axis, together with the best
        possible mean reward. The bottom row shows how often each bandit
        was chosen up to each trial.
        """
        for algo in self.algorithms:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            fig.suptitle(f"{algo.name}: learning process", fontsize=14)

            trials = np.arange(1, len(algo.rewards) + 1)
            running_avg = np.cumsum(algo.rewards) / trials
            best = max(algo.p)
            chosen = np.array(algo.chosen_bandits)

            for j, scale in enumerate(["linear", "log"]):
                ax = axes[0, j]
                ax.plot(trials, running_avg, label="Running average reward")
                ax.axhline(best, color="red", linestyle="--", label=f"Best mean = {best}")
                ax.set_xscale(scale)
                ax.set_title(f"Average reward ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Average reward")
                ax.legend(loc="lower right")
                ax.grid(alpha=0.3)

                # share of pulls of each bandit up to trial t
                ax = axes[1, j]
                for b in range(len(algo.p)):
                    share = np.cumsum(chosen == b) / trials
                    ax.plot(trials, share, linewidth=2,
                            label=f"Bandit {b + 1} (mean {algo.p[b]})")
                ax.set_xscale(scale)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Share of pulls per bandit ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Share of pulls")
                ax.legend(loc="center right", fontsize=9)
                ax.grid(alpha=0.3)

            self._finish(fig, f"plot1_{algo.name}.png")

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        """Cumulative rewards and cumulative regrets of all algorithms on one figure."""
        fig, (ax_reward, ax_regret) = plt.subplots(1, 2, figsize=(14, 5))

        for algo in self.algorithms:
            trials = np.arange(1, len(algo.rewards) + 1)
            ax_reward.plot(trials, np.cumsum(algo.rewards), label=algo.name)
            ax_regret.plot(trials, np.cumsum(algo.regrets), label=algo.name)

        ax_reward.set_title("Cumulative reward")
        ax_reward.set_xlabel("Trial")
        ax_reward.set_ylabel("Cumulative reward")
        ax_reward.legend()
        ax_reward.grid(alpha=0.3)

        ax_regret.set_title("Cumulative regret")
        ax_regret.set_xlabel("Trial")
        ax_regret.set_ylabel("Cumulative regret")
        ax_regret.legend()
        ax_regret.grid(alpha=0.3)

        self._finish(fig, "plot2_cumulative_reward_regret.png")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy algorithm where epsilon decays as ``epsilon / t``.

    With probability epsilon_t a random bandit is chosen (exploration),
    otherwise the bandit with the highest estimated mean is chosen
    (exploitation). Rewards of bandit ``i`` are drawn from
    ``N(p[i], 1 / precision)``.

    :param p: true mean reward of each bandit
    :type p: list
    :param n_trials: number of trials in the experiment
    :type n_trials: int
    :param epsilon: starting value of epsilon
    :type epsilon: float
    :param precision: precision (1 / variance) of the reward noise
    :type precision: float
    :param seed: random seed for reproducibility
    :type seed: int
    """

    name = "EpsilonGreedy"

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS, epsilon=EPSILON,
                 precision=PRECISION, seed=SEED):
        """Constructor method."""
        self.p = list(p)
        self.n_bandits = len(self.p)
        self.n_trials = n_trials
        self.epsilon = epsilon
        self.precision = precision
        self.rng = np.random.default_rng(seed)

        self.estimates = np.zeros(self.n_bandits)   # estimated mean of each bandit
        self.counts = np.zeros(self.n_bandits, dtype=int)

        self.rewards = []
        self.regrets = []
        self.chosen_bandits = []

    def __repr__(self):
        """Readable description of the object.

        :return: name and main settings of the algorithm
        :rtype: str
        """
        return (f"EpsilonGreedy(bandits={self.p}, n_trials={self.n_trials}, "
                f"epsilon={self.epsilon}, precision={self.precision})")

    def current_epsilon(self, t):
        """Epsilon used at trial ``t``.

        :param t: trial number, starting from 1
        :type t: int
        :return: ``epsilon / t``
        :rtype: float
        """
        return self.epsilon / t

    def choose_bandit(self, t):
        """Choose a bandit for trial ``t``.

        :param t: trial number, starting from 1
        :type t: int
        :return: index of the chosen bandit
        :rtype: int
        """
        if self.rng.random() < self.current_epsilon(t):
            return int(self.rng.integers(self.n_bandits))
        return int(np.argmax(self.estimates))

    def pull(self, bandit):
        """Pull a bandit and observe a reward.

        :param bandit: index of the bandit
        :type bandit: int
        :return: reward drawn from ``N(p[bandit], 1 / precision)``
        :rtype: float
        """
        return self.rng.normal(self.p[bandit], 1 / np.sqrt(self.precision))

    def update(self, bandit, reward):
        """Update the estimated mean of a bandit with a new reward (running mean).

        :param bandit: index of the bandit
        :type bandit: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[bandit] += 1
        self.estimates[bandit] += (reward - self.estimates[bandit]) / self.counts[bandit]

    def experiment(self):
        """Run the experiment for ``n_trials`` trials and store the history.

        :return: the object itself, so calls can be chained
        :rtype: EpsilonGreedy
        """
        logger.info(f"Starting experiment: {self!r}")
        best = max(self.p)

        for t in range(1, self.n_trials + 1):
            bandit = self.choose_bandit(t)
            reward = self.pull(bandit)
            self.update(bandit, reward)

            self.chosen_bandits.append(bandit)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[bandit])

            if t % 5000 == 0:
                logger.debug(f"{self.name} | trial {t} | epsilon={self.current_epsilon(t):.6f} "
                             f"| estimates={np.round(self.estimates, 3).tolist()}")

        logger.success(f"{self.name} finished {self.n_trials} trials")
        return self

    def report(self, csv_path=CSV_PATH):
        """Save the rewards to a CSV file and log the results.

        The CSV has the columns Bandit, Reward, Algorithm (one row per trial).
        If the file already exists, the rows are appended.

        :param csv_path: path of the CSV file
        :type csv_path: str
        :return: the rows written to the CSV
        :rtype: pandas.DataFrame
        """
        return _report(self, csv_path)

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision.

    Each bandit has a Normal prior on its mean. Because the precision of
    the rewards is known, the posterior is also Normal:

    * posterior precision: ``lambda_n = lambda_0 + n * tau``
    * posterior mean: ``m_n = (lambda_0 * m_0 + tau * sum(x)) / lambda_n``

    where ``tau`` is the known precision. At each trial one value is sampled
    from every posterior and the bandit with the largest sample is pulled.

    :param p: true mean reward of each bandit
    :type p: list
    :param n_trials: number of trials in the experiment
    :type n_trials: int
    :param precision: known precision (tau) of the reward noise
    :type precision: float
    :param prior_mean: prior mean ``m_0`` of every bandit
    :type prior_mean: float
    :param prior_precision: prior precision ``lambda_0`` of every bandit
    :type prior_precision: float
    :param seed: random seed for reproducibility
    :type seed: int
    """

    name = "ThompsonSampling"

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS, precision=PRECISION,
                 prior_mean=0.0, prior_precision=1.0, seed=SEED):
        """Constructor method."""
        self.p = list(p)
        self.n_bandits = len(self.p)
        self.n_trials = n_trials
        self.precision = precision
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.rng = np.random.default_rng(seed)

        self.post_mean = np.full(self.n_bandits, prior_mean, dtype=float)
        self.post_precision = np.full(self.n_bandits, prior_precision, dtype=float)
        self.sum_rewards = np.zeros(self.n_bandits)
        self.counts = np.zeros(self.n_bandits, dtype=int)

        self.rewards = []
        self.regrets = []
        self.chosen_bandits = []

    def __repr__(self):
        """Readable description of the object.

        :return: name and main settings of the algorithm
        :rtype: str
        """
        return (f"ThompsonSampling(bandits={self.p}, n_trials={self.n_trials}, "
                f"precision={self.precision}, prior_mean={self.prior_mean}, "
                f"prior_precision={self.prior_precision})")

    def sample(self):
        """Draw one sample from the posterior of every bandit.

        :return: one sampled mean per bandit
        :rtype: numpy.ndarray
        """
        return self.rng.normal(self.post_mean, 1 / np.sqrt(self.post_precision))

    def choose_bandit(self):
        """Choose the bandit with the largest posterior sample.

        :return: index of the chosen bandit
        :rtype: int
        """
        return int(np.argmax(self.sample()))

    def pull(self, bandit):
        """Pull a bandit and observe a reward.

        :param bandit: index of the bandit
        :type bandit: int
        :return: reward drawn from ``N(p[bandit], 1 / precision)``
        :rtype: float
        """
        return self.rng.normal(self.p[bandit], 1 / np.sqrt(self.precision))

    def update(self, bandit, reward):
        """Update the posterior of a bandit with a new reward.

        :param bandit: index of the bandit
        :type bandit: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[bandit] += 1
        self.sum_rewards[bandit] += reward
        self.post_precision[bandit] = self.prior_precision + self.counts[bandit] * self.precision
        self.post_mean[bandit] = (self.prior_precision * self.prior_mean
                                  + self.precision * self.sum_rewards[bandit]) \
            / self.post_precision[bandit]

    def experiment(self):
        """Run the experiment for ``n_trials`` trials and store the history.

        :return: the object itself, so calls can be chained
        :rtype: ThompsonSampling
        """
        logger.info(f"Starting experiment: {self!r}")
        best = max(self.p)

        for t in range(1, self.n_trials + 1):
            bandit = self.choose_bandit()
            reward = self.pull(bandit)
            self.update(bandit, reward)

            self.chosen_bandits.append(bandit)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[bandit])

            if t % 5000 == 0:
                logger.debug(f"{self.name} | trial {t} "
                             f"| posterior means={np.round(self.post_mean, 3).tolist()}")

        logger.success(f"{self.name} finished {self.n_trials} trials")
        return self

    def report(self, csv_path=CSV_PATH):
        """Save the rewards to a CSV file and log the results.

        The CSV has the columns Bandit, Reward, Algorithm (one row per trial).
        If the file already exists, the rows are appended.

        :param csv_path: path of the CSV file
        :type csv_path: str
        :return: the rows written to the CSV
        :rtype: pandas.DataFrame
        """
        return _report(self, csv_path)


def _report(algo, csv_path):
    """Shared report logic for both algorithms.

    :param algo: algorithm object that has already run ``experiment()``
    :type algo: EpsilonGreedy or ThompsonSampling
    :param csv_path: path of the CSV file
    :type csv_path: str
    :return: the rows written to the CSV
    :rtype: pandas.DataFrame
    """
    if not algo.rewards:
        logger.error(f"{algo.name}: no data to report, run experiment() first")
        return pd.DataFrame(columns=["Bandit", "Reward", "Algorithm"])

    df = pd.DataFrame({
        "Bandit": [b + 1 for b in algo.chosen_bandits],   # bandits numbered 1..4
        "Reward": algo.rewards,
        "Algorithm": algo.name,
    })
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    logger.info(f"{algo.name}: saved {len(df)} rows to {csv_path}")

    n = len(algo.rewards)
    cum_reward = float(np.sum(algo.rewards))
    cum_regret = float(np.sum(algo.regrets))
    pulls = {f"Bandit {b + 1}": int(c) for b, c in enumerate(algo.counts)}

    logger.info(f"{algo.name}: cumulative reward = {cum_reward:.2f}")
    logger.info(f"{algo.name}: cumulative regret = {cum_regret:.2f}")
    logger.info(f"{algo.name}: average reward = {cum_reward / n:.4f}")
    logger.info(f"{algo.name}: average regret = {cum_regret / n:.4f}")
    logger.info(f"{algo.name}: pulls per bandit = {pulls}")
    return df


def comparison(algorithms, plot_dir=PLOT_DIR):
    # think of a way to compare the performances of the two algorithms VISUALLY and
    """Compare the algorithms visually and in a summary table.

    The figure has two parts: how many times each bandit was pulled by each
    algorithm, and the running average regret on a log x-axis. A summary table
    is also logged.

    :param algorithms: algorithm objects that have already run ``experiment()``
    :type algorithms: list
    :param plot_dir: folder where the figure is saved
    :type plot_dir: str
    :return: summary table with one row per algorithm
    :rtype: pandas.DataFrame
    """
    os.makedirs(plot_dir, exist_ok=True)
    fig, (ax_pulls, ax_regret) = plt.subplots(1, 2, figsize=(14, 5))

    n_bandits = len(algorithms[0].p)
    width = 0.8 / len(algorithms)
    x = np.arange(n_bandits)
    for k, algo in enumerate(algorithms):
        ax_pulls.bar(x + k * width - 0.4 + width / 2, algo.counts, width, label=algo.name)

        trials = np.arange(1, len(algo.regrets) + 1)
        ax_regret.plot(trials, np.cumsum(algo.regrets) / trials, label=algo.name)

    ax_pulls.set_xticks(x)
    ax_pulls.set_xticklabels([f"Bandit {b + 1}\n(mean {algorithms[0].p[b]})" for b in range(n_bandits)])
    ax_pulls.set_yscale("log")
    ax_pulls.set_title("Number of pulls per bandit (log scale)")
    ax_pulls.set_ylabel("Pulls")
    ax_pulls.legend()
    ax_pulls.grid(alpha=0.3, axis="y")

    ax_regret.set_xscale("log")
    ax_regret.set_title("Average regret per trial (log x-axis)")
    ax_regret.set_xlabel("Trial")
    ax_regret.set_ylabel("Average regret")
    ax_regret.legend()
    ax_regret.grid(alpha=0.3)

    path = os.path.join(plot_dir, "comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info(f"Saved plot: {path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    best = max(algorithms[0].p)
    best_index = algorithms[0].p.index(best)
    summary = pd.DataFrame([{
        "Algorithm": a.name,
        "Cumulative reward": round(float(np.sum(a.rewards)), 2),
        "Cumulative regret": round(float(np.sum(a.regrets)), 2),
        "Best bandit pulled (%)": round(100 * a.counts[best_index] / len(a.rewards), 2),
    } for a in algorithms])
    logger.info("Comparison summary:\n" + summary.to_string(index=False))

    winner = summary.loc[summary["Cumulative regret"].idxmin(), "Algorithm"]
    logger.success(f"Lower cumulative regret: {winner}")
    return summary


if __name__=='__main__':

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
        logger.warning(f"Old {CSV_PATH} removed, a new one will be created")

    eg = EpsilonGreedy(BANDIT_REWARD).experiment()
    eg.report()

    ts = ThompsonSampling(BANDIT_REWARD).experiment()
    ts.report()

    viz = Visualization([eg, ts])
    viz.plot1()
    viz.plot2()

    comparison([eg, ts])
