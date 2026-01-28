#!/usr/bin/env python3
"""
Export WandB run charts as PNG images for documentation.

Usage:
    python training/tests/export_wandb_charts.py --run_path USER/PROJECT/RUN_ID --output_dir docs/figures
"""

import argparse
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np


def export_charts(run_path: str, output_dir: str):
    """Export key charts from a WandB run as PNG images."""
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()

    print(f"Exporting charts from: {run.url}")
    print(f"Output directory: {output_dir}")

    # Define chart groups
    chart_configs = [
        {
            "name": "loss_curves",
            "title": "NLL Loss Curves",
            "metrics": ["loss/pose_uncertainty_nll", "loss/rot_uncertainty_nll", "loss/trans_uncertainty_nll"],
            "ylabel": "NLL Loss",
        },
        {
            "name": "calibration",
            "title": "Calibration Metrics (d²)",
            "metrics": ["calibration/d2_rot_mean", "calibration/d2_trans_mean"],
            "ylabel": "d² (expect ~3)",
            "hlines": [(3.0, 'r', '--', 'Target (χ²(3) mean)')],
        },
        {
            "name": "sigma_mean",
            "title": "Sigma Evolution (σ = exp(0.5 * log_var))",
            "metrics": ["uncertainty/sigma_rot_mean", "uncertainty/sigma_trans_mean"],
            "ylabel": "σ value",
        },
        {
            "name": "sigma_percentiles",
            "title": "σ Percentiles (Uncertainty Distribution)",
            "metrics": ["diagnostic/sigma_rot_p10", "diagnostic/sigma_rot_p50", "diagnostic/sigma_rot_p90",
                       "diagnostic/sigma_trans_p10", "diagnostic/sigma_trans_p50", "diagnostic/sigma_trans_p90"],
            "ylabel": "σ value",
        },
        {
            "name": "log_var_clamp_hit",
            "title": "Log-Var Clamp Hit Rate (should be 0)",
            "metrics": ["diagnostic/log_var_rot_at_min", "diagnostic/log_var_rot_at_max",
                       "diagnostic/log_var_trans_at_min", "diagnostic/log_var_trans_at_max"],
            "ylabel": "Fraction at clamp",
        },
        {
            "name": "residual_distribution",
            "title": "Residual Distribution (p90)",
            "metrics": ["diagnostic/residual_rot_p90", "diagnostic/residual_trans_p90"],
            "ylabel": "Residual (rad / m)",
        },
        {
            "name": "scale_fitting",
            "title": "Scale Fitting Health",
            "metrics": ["scale/mean", "debug/gt_trans_norm_mean",
                       "debug/pred_trans_norm_raw_mean", "debug/pred_trans_norm_scaled_mean"],
            "ylabel": "Value",
        },
    ]

    for config in chart_configs:
        fig, ax = plt.subplots(figsize=(10, 6))

        has_data = False
        for metric in config["metrics"]:
            if metric in history.columns:
                data = history[metric].dropna()
                if len(data) > 0:
                    ax.plot(data.index, data.values, label=metric.split('/')[-1], alpha=0.8)
                    has_data = True

        if not has_data:
            plt.close(fig)
            print(f"  Skipping {config['name']} - no data")
            continue

        # Add horizontal reference lines if specified
        if "hlines" in config:
            for y, color, style, label in config["hlines"]:
                ax.axhline(y=y, color=color, linestyle=style, label=label, alpha=0.5)

        ax.set_xlabel("Step")
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f"{config['name']}.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # Export summary table as markdown
    summary_path = os.path.join(output_dir, "summary_table.md")
    with open(summary_path, 'w') as f:
        f.write("| Metric | Final Value |\n")
        f.write("|--------|-------------|\n")

        key_metrics = [
            "loss/pose_uncertainty_nll",
            "calibration/d2_rot_mean",
            "calibration/d2_trans_mean",
            "uncertainty/sigma_rot_mean",
            "uncertainty/sigma_trans_mean",
            "diagnostic/sigma_rot_p90",
            "diagnostic/sigma_trans_p90",
            "diagnostic/log_var_rot_at_min",
            "diagnostic/log_var_rot_at_max",
            "diagnostic/residual_rot_p90",
            "diagnostic/residual_trans_p90",
        ]

        for metric in key_metrics:
            if metric in history.columns:
                val = history[metric].dropna().iloc[-1] if len(history[metric].dropna()) > 0 else "N/A"
                if isinstance(val, float):
                    val = f"{val:.4f}"
                f.write(f"| `{metric}` | {val} |\n")

    print(f"  Saved: {summary_path}")
    print(f"\nDone! Add these to your markdown with:")
    print(f"  ![Loss Curves]({output_dir}/loss_curves.png)")


def main():
    parser = argparse.ArgumentParser(description='Export WandB charts as PNG')
    parser.add_argument('--run_path', type=str, required=True,
                        help='WandB run path: USER/PROJECT/RUN_ID')
    parser.add_argument('--output_dir', type=str, default='docs/figures',
                        help='Output directory for PNG files')
    args = parser.parse_args()

    export_charts(args.run_path, args.output_dir)


if __name__ == "__main__":
    main()
