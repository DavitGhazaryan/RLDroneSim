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

METRICS = [
    "z_mae",
    "xy_mae",
    "z_rmse",
    "xy_rmse",
]

REWARDS = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
]


def exact_paired_permutation_test(differences):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    observed = abs(
        np.mean(differences)
    )

    permuted = []

    for signs in itertools.product(
        [-1, 1],
        repeat=len(differences),
    ):

        signs = np.asarray(signs)

        permuted.append(
            abs(
                np.mean(
                    differences * signs
                )
            )
        )

    permuted = np.asarray(
        permuted
    )

    return float(
        np.mean(
            permuted >= observed - 1e-15
        )
    )


def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty_like(
        p_values
    )

    running_max = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):

        corrected = (
            (m - rank)
            * p_values[idx]
        )

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


# ============================================================
# Descriptive summary
# ============================================================

summary_rows = []


for algo in ["ppo", "td3"]:

    for reward in REWARDS:

        subset = df[
            (df["algo"] == algo)
            & (
                df["reward_config"]
                == reward
            )
        ]

        row = {
            "algo": algo,
            "reward": reward,
            "n_seeds":
                subset[
                    "training_seed"
                ].nunique(),
        }

        for metric in METRICS:

            column = (
                f"test_{metric}_mean"
            )

            row[
                f"{metric}_mean"
            ] = float(
                subset[column].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                subset[column].std(
                    ddof=1
                )
            )

        summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Rank reward configurations
#
# Lower error = better.
# ============================================================

for algo in ["ppo", "td3"]:

    mask = (
        summary["algo"] == algo
    )

    for metric in METRICS:

        summary.loc[
            mask,
            f"{metric}_rank",
        ] = (
            summary.loc[
                mask,
                f"{metric}_mean",
            ]
            .rank(
                method="average",
                ascending=True,
            )
        )


rank_columns = [
    f"{metric}_rank"
    for metric in METRICS
]

summary[
    "overall_rank_score"
] = summary[
    rank_columns
].mean(axis=1)


summary[
    "overall_rank"
] = (
    summary
    .groupby("algo")[
        "overall_rank_score"
    ]
    .rank(
        method="average",
        ascending=True,
    )
)


# ============================================================
# Percentage difference relative to R1
#
# Negative value = lower error than R1.
# ============================================================

for algo in ["ppo", "td3"]:

    baseline = summary[
        (summary["algo"] == algo)
        & (summary["reward"] == "R1")
    ].iloc[0]

    for metric in METRICS:

        baseline_value = baseline[
            f"{metric}_mean"
        ]

        mask = (
            summary["algo"] == algo
        )

        summary.loc[
            mask,
            f"{metric}_change_vs_R1_pct",
        ] = (
            (
                summary.loc[
                    mask,
                    f"{metric}_mean",
                ]
                - baseline_value
            )
            / baseline_value
            * 100.0
        )


# ============================================================
# Paired R1-vs-R2/R3/R4/R5 comparisons
# ============================================================

comparison_rows = []


for algo in ["ppo", "td3"]:

    algo_df = df[
        df["algo"] == algo
    ]

    for metric in METRICS:

        column = (
            f"test_{metric}_mean"
        )

        baseline = (
            algo_df[
                algo_df[
                    "reward_config"
                ] == "R1"
            ][
                [
                    "training_seed",
                    column,
                ]
            ]
            .set_index(
                "training_seed"
            )
            .sort_index()
        )

        metric_rows = []

        for reward in [
            "R2",
            "R3",
            "R4",
            "R5",
        ]:

            candidate = (
                algo_df[
                    algo_df[
                        "reward_config"
                    ] == reward
                ][
                    [
                        "training_seed",
                        column,
                    ]
                ]
                .set_index(
                    "training_seed"
                )
                .sort_index()
            )

            seeds = (
                baseline.index
                .intersection(
                    candidate.index
                )
            )

            r1_values = (
                baseline.loc[
                    seeds,
                    column,
                ]
                .to_numpy()
            )

            reward_values = (
                candidate.loc[
                    seeds,
                    column,
                ]
                .to_numpy()
            )

            # Positive:
            # R1 has larger error,
            # therefore candidate is better.
            differences = (
                r1_values
                - reward_values
            )

            p_value = (
                exact_paired_permutation_test(
                    differences
                )
            )

            metric_rows.append({
                "algo": algo,
                "metric": metric,

                "comparison":
                    f"R1_vs_{reward}",

                "reward": reward,

                "n_pairs":
                    len(seeds),

                "R1_mean":
                    float(
                        np.mean(
                            r1_values
                        )
                    ),

                "candidate_mean":
                    float(
                        np.mean(
                            reward_values
                        )
                    ),

                "mean_improvement_R1_minus_candidate":
                    float(
                        np.mean(
                            differences
                        )
                    ),

                "candidate_change_vs_R1_pct":
                    float(
                        (
                            np.mean(
                                reward_values
                            )
                            - np.mean(
                                r1_values
                            )
                        )
                        / np.mean(
                            r1_values
                        )
                        * 100.0
                    ),

                "p_exact":
                    p_value,
            })


        adjusted = (
            holm_correction(
                [
                    row["p_exact"]
                    for row in metric_rows
                ]
            )
        )

        for row, p_adj in zip(
            metric_rows,
            adjusted,
        ):

            row["p_holm"] = (
                float(p_adj)
            )

            row[
                "significant_0_05"
            ] = (
                p_adj < 0.05
            )

            comparison_rows.append(
                row
            )


comparisons = pd.DataFrame(
    comparison_rows
)


# ============================================================
# Save
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "reward_configuration_summary.csv"
)

comparison_path = (
    OUTPUT_DIR
    / "reward_vs_R1_statistics.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

comparisons.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# Print results
# ============================================================

for algo in ["ppo", "td3"]:

    print()
    print("=" * 100)
    print(
        f"{algo.upper()} — REWARD CONFIGURATION RANKING"
    )
    print("=" * 100)

    table = (
        summary[
            summary["algo"] == algo
        ][
            [
                "reward",

                "z_mae_mean",
                "xy_mae_mean",

                "z_rmse_mean",
                "xy_rmse_mean",

                "overall_rank_score",
                "overall_rank",
            ]
        ]
        .sort_values(
            "overall_rank"
        )
    )

    print(
        table.to_string(
            index=False
        )
    )


    print()
    print(
        f"{algo.upper()} — CHANGE RELATIVE TO R1 (%)"
    )

    print("-" * 100)

    table = summary[
        summary["algo"] == algo
    ][
        [
            "reward",

            "z_mae_change_vs_R1_pct",
            "xy_mae_change_vs_R1_pct",

            "z_rmse_change_vs_R1_pct",
            "xy_rmse_change_vs_R1_pct",
        ]
    ]

    print(
        table.to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("PAIRED COMPARISONS AGAINST R1")
print("=" * 100)

print(
    comparisons[
        [
            "algo",
            "metric",
            "comparison",
            "candidate_change_vs_R1_pct",
            "p_exact",
            "p_holm",
            "significant_0_05",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(summary_path)
print(comparison_path)
