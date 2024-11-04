import ee
import geemap
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Holds the validation results for a crop type."""

    crop_type: str
    mean: float
    median: float
    std: float
    count: int
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    farmer_min: float
    farmer_max: float
    within_range_pct: float


class IrrigationValidator:
    """Handles validation of irrigation model results against farmer estimates."""

    def __init__(
        self, irrigation_efficiency: float, farmer_estimates: Dict[str, List[float]]
    ):
        """
        Initialize the validator.

        Args:
            irrigation_efficiency: Efficiency factor for irrigation (0-1)
            farmer_estimates: Dictionary mapping crop types to [min, max] estimates
        """
        self.irrigation_efficiency = irrigation_efficiency
        self.farmer_estimates = farmer_estimates

    def process_feature_collection(
        self,
        feature_collection: ee.FeatureCollection,
        double_cropping_image: ee.Image,
        nutzung_column: str = "NUTZUNG",
    ) -> pd.DataFrame:
        """
        Process a GEE FeatureCollection into a pandas DataFrame.

        Args:
            feature_collection: Input FeatureCollection with ET blue values
            double_cropping_image: Image containing double cropping information
            nutzung_column: Name of the column containing crop type information

        Returns:
            Processed pandas DataFrame
        """
        # Add double cropping information
        feature_collection = self._add_double_cropping_info(
            feature_collection, double_cropping_image
        )

        # Filter for single cropping fields
        feature_collection = feature_collection.filter(
            ee.Filter.eq("isDoubleCropped", 0)
        )

        # Convert to pandas DataFrame
        df = geemap.ee_to_df(feature_collection)

        # Select and rename columns
        df = df[["ET_blue_m3_ha_yr", nutzung_column]]
        df = df.rename(columns={nutzung_column: "nutzung"})

        # Apply irrigation efficiency correction
        df["ET_blue_m3_ha_yr"] = df["ET_blue_m3_ha_yr"] / self.irrigation_efficiency

        # Filter for positive values and crop types
        df = df[df["ET_blue_m3_ha_yr"] > 0]
        df = df[df["nutzung"].isin(self.farmer_estimates.keys())]

        return df

    def _add_double_cropping_info(
        self, feature_collection: ee.FeatureCollection, double_cropping_image: ee.Image
    ) -> ee.FeatureCollection:
        """Add double cropping information to each feature."""
        filled_image = double_cropping_image.unmask(0)

        def add_double_crop_property(feature):
            median_value = (
                filled_image.select("isDoubleCropping")
                .reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=feature.geometry(),
                    scale=10,
                )
                .get("isDoubleCropping")
            )
            return feature.set("isDoubleCropped", median_value)

        return feature_collection.map(add_double_crop_property)

    def process_validation_data(
        self, df: pd.DataFrame, crops_to_validate: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Process and validate the irrigation data against farmer estimates.

        Args:
            df: DataFrame containing model results
            crops_to_validate: Optional list of crop types to validate

        Returns:
            Dictionary mapping crop types to their ValidationResult
        """
        if crops_to_validate is None:
            crops_to_validate = list(self.farmer_estimates.keys())

        validation_results = {}

        for crop in crops_to_validate:
            if crop not in self.farmer_estimates:
                continue

            crop_data = df[df["nutzung"] == crop]["ET_blue_m3_ha_yr"]

            if len(crop_data) == 0:
                continue

            stats = self._calculate_crop_statistics(
                crop_data,
                crop,
                self.farmer_estimates[crop][0],
                self.farmer_estimates[crop][1],
            )

            validation_results[crop] = stats

        return validation_results

    def _calculate_crop_statistics(
        self, crop_data: pd.Series, crop_type: str, farmer_min: float, farmer_max: float
    ) -> ValidationResult:
        """Calculate validation statistics for a single crop type."""
        within_range = (
            (crop_data >= farmer_min) & (crop_data <= farmer_max)
        ).mean() * 100

        return ValidationResult(
            crop_type=crop_type,
            mean=crop_data.mean(),
            median=crop_data.median(),
            std=crop_data.std(),
            count=len(crop_data),
            min_value=crop_data.min(),
            max_value=crop_data.max(),
            percentile_25=crop_data.quantile(0.25),
            percentile_75=crop_data.quantile(0.75),
            farmer_min=farmer_min,
            farmer_max=farmer_max,
            within_range_pct=within_range,
        )

