import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ANALYSIS_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis"
)

CANDIDATES_PATH = (
    ANALYSIS_DIR / "candidate_selection_scores.csv"
)

SELECTED_PATH = (
    ANALYSIS_DIR / "final_selected_models.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Representative configurations
# ============================================================

PPO_REWARD = "R2"
TD3_REWARD = "R2"

SEEDS = [101, 202, 303, 404, 505]


# ============================================================
# Load data
# ============================================================

candidates = pd.read_csv(CANDIDATES_PATH)
selected = pd.read_csv(SELECTED_PATH)

print("Candidate rows:", len(candidates))
print("Selected models:", len(selected))


# ============================================================
# Plot helper
# ============================================================

def plot_configuration(
    ax,
    algo,
    reward,
    panel_title,
):

    config_candidates = candidates[
        (candidates["algo"].str.lower() == algo.lower())
        & (candidates["reward_config"] == reward)
    ].copy()

    config_selected = selected[
        (selected["algo"].str.lower() == algo.lower())
        & (selected["reward_config"] == reward)
    ].copy()

    print()
    print("=" * 70)
    print(f"{algo.upper()} {reward}")
    print("=" * 70)

    print(
        "Number of candidate checkpoints:",
        len(config_candidates),
    )

    print(
        "Candidates per seed:"
    )

    print(
        config_candidates[
            "training_seed"
        ].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Plot candidates for each training seed
    # --------------------------------------------------------

    for seed in SEEDS:

        seed_candidates = config_candidates[
            config_candidates["training_seed"] == seed
        ]

        if seed_candidates.empty:
            continue

        scatter = ax.scatter(
            seed_candidates["sel_z_mae_mean"],
            seed_candidates["sel_xy_mae_mean"],
            s=65,
            alpha=0.75,
            label=f"Seed {seed}",
        )

        # ----------------------------------------------------
        # Highlight selected model for this seed
        # ----------------------------------------------------

        seed_selected = config_selected[
            config_selected["training_seed"] == seed
        ]

        if len(seed_selected) != 1:
            print(
                f"Warning: expected one selected model "
                f"for seed {seed}, found {len(seed_selected)}"
            )
            continue

        selected_row = seed_selected.iloc[0]

        ax.scatter(
            selected_row["sel_z_mae_mean"],
            selected_row["sel_xy_mae_mean"],
            marker="*",
            s=260,
            color=scatter.get_facecolor()[0],
            edgecolors="black",
            linewidths=1.2,
            zorder=10,
            label="Selected checkpoint" if seed == SEEDS[0] else None,
        )

        checkpoint = str(
            selected_row["checkpoint"]
        )

        if checkpoint.endswith(".0"):
            checkpoint = checkpoint[:-2]

        # Fixed label positions inside the axes
        label_positions = {
            101: (0.25, 0.42),
            202: (0.25, 0.32),
            303: (0.25, 0.22),
            404: (0.25, 0.52),
            505: (0.25, 0.05),
        }

        text_x, text_y = label_positions[seed]

        ax.annotate(
            f"{seed}: {checkpoint}",
            xy=(
                selected_row["sel_z_mae_mean"],
                selected_row["sel_xy_mae_mean"],
            ),
            xycoords="data",

            xytext=(text_x, text_y),
            textcoords="axes fraction",

            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",

            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.8,
                shrinkA=3,
                shrinkB=4,
            ),

            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor="0.7",
            ),
        )
    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{panel_title} {algo.upper()} - {reward}"
    )

    ax.set_xlabel(
        "Altitude MAE (m)"
    )

    ax.set_ylabel(
        "Horizontal-position MAE (m)"
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        title="Training seed / Selection",
    )


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5),
)


plot_configuration(
    axes[0],
    algo="ppo",
    reward=PPO_REWARD,
    panel_title="(a)",
)

plot_configuration(
    axes[1],
    algo="td3",
    reward=TD3_REWARD,
    panel_title="(b)",
)


plt.tight_layout()


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIR
    / "checkpoint_candidate_selection.png"
)

pdf_path = (
    OUTPUT_DIR
    / "checkpoint_candidate_selection.pdf"
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


print()
print("Saved:")
print(png_path)
print(pdf_path)