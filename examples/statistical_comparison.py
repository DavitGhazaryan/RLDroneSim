import itertools
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis/final_selected_models.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Primary/secondary control metrics
METRICS = [
    "z_mae",
    "xy_mae",
    "z_rmse",
    "xy_rmse",
]


def exact_paired_permutation_test(differences):
    """
    Exact two-sided paired sign-flip permutation test.

    H0: the paired differences are centered around zero.

    differences are defined here as:
        TD3 - PPO

    Positive difference means PPO has lower error.
    """

    differences = np.asarray(differences, dtype=float)

    observed = abs(np.mean(differences))

    permuted = []

    for signs in itertools.product([-1, 1], repeat=len(differences)):
        signs = np.asarray(signs)
        permuted.append(
            abs(np.mean(differences * signs))
        )

    permuted = np.asarray(permuted)

    p_value = np.mean(
        permuted >= observed - 1e-15
    )

    return float(p_value)


def holm_correction(p_values):
    """
    Holm multiple-comparison correction.
    """

    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running_max = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        corrected = (m - rank) * p_values[idx]

        running_max = max(
            running_max,
            corrected,
        )

        adjusted[idx] = min(
            running_max,
            1.0,
        )

    return adjusted


df = pd.read_csv(INPUT)

rows = []


for metric in METRICS:

    column = f"test_{metric}_mean"

    metric_rows = []

    for reward in sorted(
        df["reward_config"].unique()
    ):

        subset = df[
            df["reward_config"] == reward
        ]

        ppo = (
            subset[
                subset["algo"] == "ppo"
            ][
                ["training_seed", column]
            ]
            .set_index("training_seed")
            .sort_index()
        )

        td3 = (
            subset[
                subset["algo"] == "td3"
            ][
                ["training_seed", column]
            ]
            .set_index("training_seed")
            .sort_index()
        )

        common_seeds = (
            ppo.index
            .intersection(td3.index)
        )

        ppo_values = (
            ppo.loc[
                common_seeds,
                column
            ]
            .to_numpy()
        )

        td3_values = (
            td3.loc[
                common_seeds,
                column
            ]
            .to_numpy()
        )

        # Positive = TD3 error is larger than PPO error
        differences = (
            td3_values - ppo_values
        )

        p_value = (
            exact_paired_permutation_test(
                differences
            )
        )

        mean_diff = float(
            np.mean(differences)
        )

        median_diff = float(
            np.median(differences)
        )

        mean_ppo = float(
            np.mean(ppo_values)
        )

        mean_td3 = float(
            np.mean(td3_values)
        )

        ratio = (
            mean_td3 / mean_ppo
            if mean_ppo != 0
            else np.nan
        )

        # Paired standardized effect size
        diff_std = np.std(
            differences,
            ddof=1,
        )

        cohens_dz = (
            mean_diff / diff_std
            if diff_std > 0
            else np.inf
        )

        metric_rows.append({
            "metric": metric,
            "reward": reward,
            "n_pairs": len(common_seeds),

            "ppo_mean": mean_ppo,
            "ppo_std": float(
                np.std(
                    ppo_values,
                    ddof=1,
                )
            ),

            "td3_mean": mean_td3,
            "td3_std": float(
                np.std(
                    td3_values,
                    ddof=1,
                )
            ),

            "mean_difference_td3_minus_ppo":
                mean_diff,

            "median_difference_td3_minus_ppo":
                median_diff,

            "td3_to_ppo_ratio":
                ratio,

            "cohens_dz":
                cohens_dz,

            "p_exact":
                p_value,
        })


    # Correct the 5 reward comparisons
    # separately for each metric.
    raw_p = [
        row["p_exact"]
        for row in metric_rows
    ]

    adjusted = holm_correction(
        raw_p
    )

    for row, p_adj in zip(
        metric_rows,
        adjusted,
    ):

        row["p_holm"] = float(
            p_adj
        )

        row["significant_0_05"] = (
            p_adj < 0.05
        )

        rows.append(row)


results = pd.DataFrame(rows)

output_path = (
    OUTPUT_DIR
    / "ppo_vs_td3_statistics.csv"
)

results.to_csv(
    output_path,
    index=False,
)


print("=" * 100)
print("PPO vs TD3 — PAIRED SEED-LEVEL COMPARISON")
print("=" * 100)

for metric in METRICS:

    print()
    print("-" * 100)
    print(metric.upper())
    print("-" * 100)

    table = results[
        results["metric"] == metric
    ][
        [
            "reward",
            "ppo_mean",
            "td3_mean",
            "mean_difference_td3_minus_ppo",
            "td3_to_ppo_ratio",
            "cohens_dz",
            "p_exact",
            "p_holm",
            "significant_0_05",
        ]
    ]

    print(
        table.to_string(
            index=False
        )
    )


print()
print("Saved:")
print(output_path)
