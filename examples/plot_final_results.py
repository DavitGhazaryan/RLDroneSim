from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis/final_selected_models.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis/figures"
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


def calculate_summary(metric):

    column = f"test_{metric}_mean"

    summary = (
        df
        .groupby(
            ["reward_config", "algo"]
        )[column]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary


def plot_metric(metric, ylabel):

    summary = calculate_summary(
        metric
    )

    x = np.arange(
        len(REWARDS)
    )

    width = 0.35

    ppo_means = []
    ppo_stds = []

    td3_means = []
    td3_stds = []


    for reward in REWARDS:

        ppo_row = summary[
            (summary["reward_config"] == reward)
            & (summary["algo"] == "ppo")
        ].iloc[0]

        td3_row = summary[
            (summary["reward_config"] == reward)
            & (summary["algo"] == "td3")
        ].iloc[0]

        ppo_means.append(
            ppo_row["mean"]
        )

        ppo_stds.append(
            ppo_row["std"]
        )

        td3_means.append(
            td3_row["mean"]
        )

        td3_stds.append(
            td3_row["std"]
        )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        x - width / 2,
        ppo_means,
        width,
        yerr=ppo_stds,
        capsize=4,
        label="PPO",
    )

    ax.bar(
        x + width / 2,
        td3_means,
        width,
        yerr=td3_stds,
        capsize=4,
        label="TD3",
    )


    ax.set_xlabel(
        "Reward Configuration"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        REWARDS
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()


    png_path = (
        OUTPUT_DIR
        / f"{metric}_comparison.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{metric}_comparison.pdf"
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

    print(
        f"Saved: {pdf_path}"
    )


for metric, ylabel in METRICS.items():

    plot_metric(
        metric,
        ylabel,
    )


print()
print(
    "All figures saved to:"
)

print(OUTPUT_DIR)
