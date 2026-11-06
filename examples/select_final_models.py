import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

EPISODES_PATH = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_50ep_episodes.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Evaluation split
# ============================================================

SELECTION_SEEDS = list(range(1001, 1026))
HOLDOUT_SEEDS = list(range(1026, 1051))


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(EPISODES_PATH)


def normalize_checkpoint(value):

    value = str(value).strip()

    if value.lower() == "best":
        return "best"

    return str(int(float(value)))


df["checkpoint"] = (
    df["checkpoint"]
    .apply(normalize_checkpoint)
)


print("=" * 80)
print("FINAL MODEL SELECTION")
print("=" * 80)

print(f"Episode rows: {len(df)}")

print(
    "Candidates:",
    df[
        [
            "experiment",
            "algo",
            "training_seed",
            "checkpoint",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "Training runs:",
    df[
        [
            "experiment",
            "reward_config",
            "algo",
            "training_seed",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)


# ============================================================
# Candidate identifiers
# ============================================================

candidate_cols = [
    "experiment",
    "reward_config",
    "algo",
    "training_seed",
    "checkpoint",
    "selection_reason",
]

run_cols = [
    "experiment",
    "reward_config",
    "algo",
    "training_seed",
]


# ============================================================
# Metrics
# ============================================================

physical_metrics = [
    "z_mae",
    "xy_mae",
    "z_rmse",
    "xy_rmse",
    "max_z_error",
    "max_xy_error",
]


# ============================================================
# Split evaluation episodes
# ============================================================

selection_df = df[
    df["eval_seed"].isin(
        SELECTION_SEEDS
    )
].copy()

holdout_df = df[
    df["eval_seed"].isin(
        HOLDOUT_SEEDS
    )
].copy()


print()
print(
    "Selection episodes:",
    len(selection_df),
)

print(
    "Holdout episodes:",
    len(holdout_df),
)


# ============================================================
# Aggregate selection episodes
# ============================================================

selection_agg = {}


selection_agg[
    "selection_episodes"
] = (
    "episode",
    "count",
)


for metric in physical_metrics:

    selection_agg[
        f"sel_{metric}_mean"
    ] = (
        metric,
        "mean",
    )

    selection_agg[
        f"sel_{metric}_std"
    ] = (
        metric,
        "std",
    )


selection_scores = (
    selection_df
    .groupby(
        candidate_cols,
        as_index=False,
    )
    .agg(**selection_agg)
)


# ============================================================
# Check that every candidate has 25 selection episodes
# ============================================================

print()
print(
    "Selection episode counts:"
)

print(
    selection_scores[
        "selection_episodes"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# Balanced checkpoint score
#
# Lower percentile rank = better.
#
# Means represent accuracy.
# Standard deviations of MAE represent consistency.
# ============================================================

score_metrics = [
    "sel_z_mae_mean",
    "sel_xy_mae_mean",

    "sel_z_rmse_mean",
    "sel_xy_rmse_mean",

    "sel_max_z_error_mean",
    "sel_max_xy_error_mean",

    "sel_z_mae_std",
    "sel_xy_mae_std",
]


rank_cols = []


for metric in score_metrics:

    rank_col = (
        f"rank_{metric}"
    )

    selection_scores[
        rank_col
    ] = (
        selection_scores
        .groupby(
            run_cols
        )[metric]
        .rank(
            method="average",
            ascending=True,
            pct=True,
        )
    )

    rank_cols.append(
        rank_col
    )


selection_scores[
    "balanced_score"
] = (
    selection_scores[
        rank_cols
    ]
    .mean(axis=1)
)


# ============================================================
# Deterministic tie breaking
# ============================================================

selection_scores[
    "checkpoint_numeric"
] = pd.to_numeric(
    selection_scores[
        "checkpoint"
    ],
    errors="coerce",
)


sort_cols = (
    run_cols
    + [
        "balanced_score",

        "sel_z_mae_mean",
        "sel_xy_mae_mean",

        "sel_z_rmse_mean",
        "sel_xy_rmse_mean",

        "sel_max_z_error_mean",
        "sel_max_xy_error_mean",

        "checkpoint_numeric",
    ]
)


selection_scores = (
    selection_scores
    .sort_values(
        sort_cols,
        na_position="last",
    )
)


# ============================================================
# Select one checkpoint per training run
# ============================================================

final_models = (
    selection_scores
    .groupby(
        run_cols,
        group_keys=False,
    )
    .head(1)
    .copy()
)


print()
print(
    "Final selected models:",
    len(final_models),
)


# ============================================================
# Aggregate holdout episodes
# ============================================================

holdout_agg = {}


holdout_agg[
    "holdout_episodes"
] = (
    "episode",
    "count",
)


for metric in physical_metrics:

    holdout_agg[
        f"test_{metric}_mean"
    ] = (
        metric,
        "mean",
    )

    holdout_agg[
        f"test_{metric}_std"
    ] = (
        metric,
        "std",
    )


# Also retain reward and length only as descriptive metrics

for metric in [
    "return",
    "length",
]:

    holdout_agg[
        f"test_{metric}_mean"
    ] = (
        metric,
        "mean",
    )

    holdout_agg[
        f"test_{metric}_std"
    ] = (
        metric,
        "std",
    )


holdout_scores = (
    holdout_df
    .groupby(
        candidate_cols,
        as_index=False,
    )
    .agg(**holdout_agg)
)


# ============================================================
# Merge unseen test performance for the chosen checkpoint
# ============================================================

final_models = (
    final_models
    .merge(
        holdout_scores,
        on=candidate_cols,
        how="left",
    )
)


print()
print(
    "Holdout episode counts "
    "for selected models:"
)

print(
    final_models[
        "holdout_episodes"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# Preserve selected holdout episodes
# ============================================================

selected_keys = (
    final_models[
        candidate_cols
    ]
    .drop_duplicates()
)


selected_holdout = (
    holdout_df
    .merge(
        selected_keys,
        on=candidate_cols,
        how="inner",
    )
)


# ============================================================
# Aggregate across the 5 independent training seeds
#
# IMPORTANT:
# Statistics here are calculated across the 5 training runs,
# not by treating all episodes as independent training runs.
# ============================================================

group_cols = [
    "reward_config",
    "algo",
]


group_rows = []


for keys, group in (
    final_models
    .groupby(group_cols)
):

    reward_config, algo = keys

    row = {
        "reward": reward_config,
        "algo": algo,
        "n_training_seeds":
            group[
                "training_seed"
            ].nunique(),
    }


    for metric in physical_metrics:

        column = (
            f"test_{metric}_mean"
        )

        row[
            f"{metric}_mean_across_seeds"
        ] = float(
            group[column].mean()
        )

        row[
            f"{metric}_std_across_seeds"
        ] = float(
            group[column].std(
                ddof=1
            )
        )


    group_rows.append(row)


group_summary = pd.DataFrame(
    group_rows
)


# ============================================================
# Save results
# ============================================================

scores_path = (
    OUTPUT_DIR
    / "candidate_selection_scores.csv"
)

models_path = (
    OUTPUT_DIR
    / "final_selected_models.csv"
)

episodes_path = (
    OUTPUT_DIR
    / "final_selected_holdout_episodes.csv"
)

group_path = (
    OUTPUT_DIR
    / "final_group_summary.csv"
)


selection_scores.to_csv(
    scores_path,
    index=False,
)

final_models.to_csv(
    models_path,
    index=False,
)

selected_holdout.to_csv(
    episodes_path,
    index=False,
)

group_summary.to_csv(
    group_path,
    index=False,
)


# ============================================================
# Print selected models
# ============================================================

print()
print("=" * 80)
print("SELECTED CHECKPOINTS")
print("=" * 80)

print(
    final_models[
        [
            "reward_config",
            "algo",
            "training_seed",
            "checkpoint",
            "selection_reason",
            "balanced_score",
            "test_z_mae_mean",
            "test_xy_mae_mean",
        ]
    ]
    .sort_values(
        [
            "reward_config",
            "algo",
            "training_seed",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("GROUP SUMMARY")
print("=" * 80)

print(
    group_summary
    .sort_values(
        [
            "reward",
            "algo",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(models_path)
print(group_path)
print(episodes_path)
print(scores_path)
