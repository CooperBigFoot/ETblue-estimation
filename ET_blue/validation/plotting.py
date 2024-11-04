from typing import Dict, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from .irrigation_validation import ValidationResult


class ValidationPlotter:
    """Handles plotting of irrigation validation results."""

    def __init__(self, style_config: Optional[dict] = None):
        """
        Initialize the plotter.

        Args:
            style_config: Optional dictionary of matplotlib style settings
        """
        self.style_config = style_config or {
            "figure.figsize": (10, 6),
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }

    def _setup_plot_style(self):
        """Apply plot style settings."""
        for key, value in self.style_config.items():
            plt.rcParams[key] = value

    def create_validation_plots(
        self,
        df: pd.DataFrame,
        validation_results: Dict[str, ValidationResult],
        output_dir: Path,
        dpi: int = 300,
    ) -> None:
        """
        Create validation plots for each crop type.

        Args:
            df: DataFrame containing the model results
            validation_results: Dictionary of ValidationResults by crop type
            output_dir: Directory to save plots
            dpi: Resolution of output plots
        """
        self._setup_plot_style()

        for crop_type, result in validation_results.items():
            self._create_single_crop_plot(
                df=df, result=result, output_dir=output_dir, dpi=dpi
            )

    def _create_single_crop_plot(
        self, df: pd.DataFrame, result: ValidationResult, output_dir: Path, dpi: int
    ) -> None:
        """Create and save a validation plot for a single crop type."""
        crop_data = df[df["nutzung"] == result.crop_type]["ET_blue_m3_ha_yr"]

        fig, ax = plt.subplots()

        # Create histogram with KDE
        sns.histplot(crop_data, kde=True, ax=ax)

        # Add farmer estimate range
        ax.axvline(
            result.farmer_min,
            color="r",
            linestyle="--",
            label="Farmer's estimate range",
        )
        ax.axvline(result.farmer_max, color="r", linestyle="--")

        # Add mean line
        ax.axvline(
            result.mean,
            color="g",
            linestyle="-",
            label=f"Mean: {result.mean:.0f} m³/ha/yr",
        )

        # Add median line
        ax.axvline(
            result.median,
            color="b",
            linestyle=":",
            label=f"Median: {result.median:.0f} m³/ha/yr",
        )

        # Customize plot
        ax.set_xlabel("ET Blue (m³/ha/yr)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Irrigation Validation - {result.crop_type}\n"
            f"{result.within_range_pct:.1f}% within farmer estimates"
        )

        # Add legend
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

        # Adjust layout and save
        plt.tight_layout()
        output_path = (
            output_dir / f'validation_{result.crop_type.replace(" ", "_")}.png'
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close()
