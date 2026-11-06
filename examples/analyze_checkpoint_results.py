import pandas as pd
import numpy as np
from pathlib import Path


INPUT = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "all_experiments/checkpoint_summary.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "all_experiments/analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

print(f"Loaded {len(df)} checkpoint evaluations.")


# ---------------------------------------------------------
# Separate systematic checkpoints from best_model
# ---------------------------------------------------------

fixed = df[df["checkpoint"] != "best"].copy()
best_models = df[df["checkpoint"] == "best"].copy()

fixed["checkpoint"] = fixed["checkpoint"].astype(int)


# ---------------------------------------------------------
# Metrics used for balanced checkpoint selection
# Lower is better for all of them.
# ---------------------------------------------------------

ranking_metrics = [
    "z_mae",
    "xy_mae",
    "z_rmse",
    "xy_rmse",
    "max_z_error",
    "max_xy_error",
]


group_cols = [
    "experiment",
    "reward",
    "algo",
    "seed",
]


# ---------------------------------------------------------
# Calculate within-run percentile ranks
# ---------------------------------------------------------

for metric in ranking_metrics:

    rank_col = f"rank_{metric}"

    fixed[rank_col] = (
        fixed
        .groupby(group_cols)[metric]
        .rank(
            method="average",
            ascending=True,
            pct=True,
        )
    )


rank_cols = [
    f"rank_{metric}"
    for metric in ranking_metrics
]


fixed["balanced_score"] = fixed[rank_cols].mean(axis=1)


# Lower balanced_score = better overall physical performance


# ---------------------------------------------------------
# Select candidates for every training run
# ---------------------------------------------------------

selected_rows = []


for keys, group in fixed.groupby(group_cols):

    experiment, reward, algo, seed = keys

    candidates = []


    # 1. Best altitude checkpoint
    altitude_row = (
    group
    .sort_values(
        [
            "z_mae",
            "z_rmse",
            "max_z_error",
            "xy_mae",
            "checkpoint",
        ]
    )
    .iloc[0]
    .copy()
)

    altitude_row["selection_reason"] = "best_z_mae"

    candidates.append(altitude_row)


    # 2. Best XY checkpoint
    xy_min = group["xy_mae"].min()

    xy_candidates = group[
        np.isclose(
            group["xy_mae"],
            xy_min,
            rtol=0,
            atol=1e-12,
        )
    ]

    if len(xy_candidates) == 1:

        xy_row = xy_candidates.iloc[0].copy()

        xy_row["selection_reason"] = "best_xy_mae"

        candidates.append(xy_row)


    # 3. Best balanced checkpoint
    balanced_row = group.loc[
        group["balanced_score"].idxmin()
    ].copy()

    balanced_row["selection_reason"] = "best_balanced"

    candidates.append(balanced_row)


    # -----------------------------------------------------
    # Deduplicate fixed checkpoints
    # If the same checkpoint wins several criteria,
    # combine the reasons.
    # -----------------------------------------------------

    candidate_dict = {}

    for row in candidates:

        checkpoint = int(row["checkpoint"])

        if checkpoint not in candidate_dict:
            candidate_dict[checkpoint] = row

        else:
            previous_reason = (
                candidate_dict[checkpoint][
                    "selection_reason"
                ]
            )

            candidate_dict[checkpoint][
                "selection_reason"
            ] = (
                previous_reason
                + "+"
                + row["selection_reason"]
            )


    selected_rows.extend(
        candidate_dict.values()
    )


    # -----------------------------------------------------
    # 4. Add best_model as independent reference
    # -----------------------------------------------------

    best_match = best_models[
        (best_models["experiment"] == experiment)
        & (best_models["reward"] == reward)
        & (best_models["algo"] == algo)
        & (best_models["seed"] == seed)
    ]

    if len(best_match) == 1:

        best_row = best_match.iloc[0].copy()

        best_row["selection_reason"] = (
            "training_best_model"
        )

        best_row["balanced_score"] = None

        selected_rows.append(best_row)


# ---------------------------------------------------------
# Create shortlist
# ---------------------------------------------------------

shortlist = pd.DataFrame(selected_rows)


columns = [
    "experiment",
    "reward",
    "algo",
    "seed",
    "checkpoint",
    "selection_reason",

    "balanced_score",

    "z_mae",
    "xy_mae",

    "z_rmse",
    "xy_rmse",

    "max_z_error",
    "max_xy_error",

    "avg_reward",
    "avg_length",
]


shortlist = shortlist[columns]


shortlist = shortlist.sort_values(
    [
        "reward",
        "algo",
        "seed",
        "checkpoint",
    ]
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

shortlist_path = (
    OUTPUT_DIR /
    "checkpoint_shortlist.csv"
)

shortlist.to_csv(
    shortlist_path,
    index=False,
)


rankings_path = (
    OUTPUT_DIR /
    "all_fixed_checkpoint_rankings.csv"
)

fixed.to_csv(
    rankings_path,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 80)
print("CHECKPOINT SHORTLIST")
print("=" * 80)

print(
    f"Training runs: "
    f"{df[group_cols].drop_duplicates().shape[0]}"
)

print(
    f"Shortlisted checkpoints: "
    f"{len(shortlist)}"
)

print()
print("Selection reasons:")

print(
    shortlist["selection_reason"]
    .value_counts()
)

print()
print(f"Shortlist saved to:")
print(shortlist_path)

print()
print(f"Full checkpoint rankings saved to:")
print(rankings_path)
