from pathlib import Path

import pandas as pd


INPUT = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis/final_group_summary.csv"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "final_50ep/final_analysis"
)

CSV_OUTPUT = OUTPUT_DIR / "final_results_table.csv"
LATEX_OUTPUT = OUTPUT_DIR / "final_results_table.tex"


df = pd.read_csv(INPUT)


def format_metric(row, metric):

    mean_col = f"{metric}_mean_across_seeds"
    std_col = f"{metric}_std_across_seeds"

    return "{:.5f} ± {:.5f}".format(
        row[mean_col],
        row[std_col],
    )


rows = []


for _, row in df.iterrows():

    rows.append(
        {
            "Reward": row["reward"],
            "Algorithm": row["algo"].upper(),

            "Z MAE (m)": format_metric(
                row,
                "z_mae",
            ),

            "XY MAE (m)": format_metric(
                row,
                "xy_mae",
            ),

            "Z RMSE (m)": format_metric(
                row,
                "z_rmse",
            ),

            "XY RMSE (m)": format_metric(
                row,
                "xy_rmse",
            ),
        }
    )


table = pd.DataFrame(rows)


# ============================================================
# Save CSV
# ============================================================

table.to_csv(
    CSV_OUTPUT,
    index=False,
)


# ============================================================
# Create LaTeX table
# ============================================================

latex_lines = []

latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")

latex_lines.append(
    r"\caption{Holdout evaluation performance of PPO and TD3 "
    r"across the five reward configurations. Values are reported "
    r"as mean $\pm$ standard deviation across five independent "
    r"training seeds. Lower values are better.}"
)

latex_lines.append(
    r"\label{tab:final_results}"
)

latex_lines.append(
    r"\begin{tabular}{llcccc}"
)

latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"Reward & Algorithm & Z MAE & XY MAE & Z RMSE & XY RMSE \\"
)

latex_lines.append(
    r" & & (m) & (m) & (m) & (m) \\"
)

latex_lines.append(
    r"\hline"
)


for _, row in table.iterrows():

    z_mae = row["Z MAE (m)"].replace(
        "±",
        r"$\pm$",
    )

    xy_mae = row["XY MAE (m)"].replace(
        "±",
        r"$\pm$",
    )

    z_rmse = row["Z RMSE (m)"].replace(
        "±",
        r"$\pm$",
    )

    xy_rmse = row["XY RMSE (m)"].replace(
        "±",
        r"$\pm$",
    )

    line = "{} & {} & {} & {} & {} & {} \\\\".format(
        row["Reward"],
        row["Algorithm"],
        z_mae,
        xy_mae,
        z_rmse,
        xy_rmse,
    )

    latex_lines.append(line)


latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"\end{tabular}"
)

latex_lines.append(
    r"\end{table}"
)


LATEX_OUTPUT.write_text(
    "\n".join(latex_lines)
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("FINAL RESULTS TABLE")
print("=" * 100)

print(
    table.to_string(
        index=False
    )
)

print()
print("Saved:")
print(CSV_OUTPUT)
print(LATEX_OUTPUT)