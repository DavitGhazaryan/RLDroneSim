
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "/home/pid_rl/rl_training/evaluation_results/all_experiments/checkpoint_summary.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load checkpoint-screening results
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# Keep only numbered checkpoints
# ============================================================
# The saved "best" model is excluded because it does not
# correspond to a unique training step.

df["checkpoint_numeric"] = pd.to_numeric(
    df["checkpoint"],
    errors="coerce",
)

screening = df[
    df["checkpoint_numeric"].notna()
].copy()

screening["checkpoint_numeric"] = (
    screening["checkpoint_numeric"].astype(int)
)

# Optional safety check
screening = screening[
    (screening["checkpoint_numeric"] >= 400_000)
    & (screening["checkpoint_numeric"] <= 1_000_000)
].copy()


# ============================================================
# Aggregate across the five independent training seeds
# ============================================================

summary = (
    screening
    .groupby(
        [
            "reward",
            "algo",
            "checkpoint_numeric",
        ],
        as_index=False,
    )
    .agg(
        z_mae_mean=("z_mae", "mean"),
        z_mae_std=("z_mae", "std"),
        n_seeds=("seed", "nunique"),
    )
)


print("\nNumber of seeds at each point:")
print(summary["n_seeds"].value_counts().sort_index())


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharex=True,
)

reward_configs = ["R1", "R2", "R3", "R4", "R5"]
algorithms = ["ppo", "td3"]

panel_titles = {
    "ppo": "(a) PPO",
    "td3": "(b) TD3",
}


for ax, algo in zip(axes, algorithms):

    algo_df = summary[
        summary["algo"].str.lower() == algo
    ]

    for reward in reward_configs:

        reward_df = (
            algo_df[
                algo_df["reward"] == reward
            ]
            .sort_values("checkpoint_numeric")
        )

        if reward_df.empty:
            continue

        x = (
            reward_df["checkpoint_numeric"]
            .to_numpy()
            / 1000.0
        )

        mean = reward_df[
            "z_mae_mean"
        ].to_numpy()

        std = reward_df[
            "z_mae_std"
        ].fillna(0).to_numpy()

        line = ax.plot(
            x,
            mean,
            linewidth=1.8,
            label=reward,
        )[0]

        ax.fill_between(
            x,
            np.maximum(mean - std, 0),
            mean + std,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_title(panel_titles[algo])

    ax.set_xlabel(
        "Training steps (×1000)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        title="Reward",
        frameon=False,
    )


axes[0].set_ylabel(
    "Altitude MAE (m)"
)


# Separate y-axis scales are intentionally used because
# PPO and TD3 operate at substantially different error levels.
axes[0].set_ylim(bottom=0)

# Limit TD3 axis to improve visibility of the main performance range.
# Values above 0.2 m are clipped from the visualization but remain
# unchanged in the underlying data and analysis.
axes[1].set_ylim(0, 0.2)


plt.tight_layout()


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIR
    / "checkpoint_screening_z_mae.png"
)

pdf_path = (
    OUTPUT_DIR
    / "checkpoint_screening_z_mae.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close()


print(f"\nSaved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")