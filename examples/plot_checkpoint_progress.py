from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "all_experiments/checkpoint_summary.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis/checkpoint_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REWARDS = ["R1", "R2", "R3", "R4", "R5"]

METRICS = {
    "z_mae": "Z MAE (m)",
    "xy_mae": "XY MAE (m)",
    "z_rmse": "Z RMSE (m)",
    "xy_rmse": "XY RMSE (m)",
}


df = pd.read_csv(INPUT)


# ------------------------------------------------------------
# Keep only fixed checkpoints
# ------------------------------------------------------------

df = df[
    df["checkpoint"].astype(str) != "best"
].copy()

df["checkpoint"] = pd.to_numeric(
    df["checkpoint"],
    errors="coerce",
)

df = df.dropna(
    subset=["checkpoint"]
)

df["checkpoint"] = (
    df["checkpoint"]
    .astype(int)
)


# ------------------------------------------------------------
# Aggregate across the five training seeds
# ------------------------------------------------------------

summary = (
    df
    .groupby(
        [
            "reward",
            "algo",
            "checkpoint",
        ]
    )
    .agg(
        z_mae_mean=("z_mae", "mean"),
        z_mae_std=("z_mae", "std"),

        xy_mae_mean=("xy_mae", "mean"),
        xy_mae_std=("xy_mae", "std"),

        z_rmse_mean=("z_rmse", "mean"),
        z_rmse_std=("z_rmse", "std"),

        xy_rmse_mean=("xy_rmse", "mean"),
        xy_rmse_std=("xy_rmse", "std"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Plot one figure for each algorithm and metric
# ------------------------------------------------------------

for algo in ["ppo", "td3"]:

    for metric, ylabel in METRICS.items():

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for reward in REWARDS:

            data = summary[
                (summary["algo"] == algo)
                & (summary["reward"] == reward)
            ].sort_values(
                "checkpoint"
            )

            x = (
                data["checkpoint"]
                / 1000
            )

            mean = data[
                f"{metric}_mean"
            ]

            std = data[
                f"{metric}_std"
            ]

            ax.plot(
                x,
                mean,
                label=reward,
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.15,
            )


        ax.set_xlabel(
            "Training Steps (thousands)"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{algo.upper()} – {ylabel}"
        )

        ax.legend(
            title="Reward"
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()


        png_path = (
            OUTPUT_DIR
            / f"{algo}_{metric}_vs_training_steps.png"
        )

        pdf_path = (
            OUTPUT_DIR
            / f"{algo}_{metric}_vs_training_steps.pdf"
        )


        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {png_path}"
        )


print()
print(
    "Checkpoint progress figures saved to:"
)
print(OUTPUT_DIR)
