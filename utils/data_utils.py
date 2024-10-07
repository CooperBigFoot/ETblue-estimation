import pandas as pd
import numpy as np


def aggregate_time_series(
    daily_data: pd.DataFrame,
    validation_dates: pd.DataFrame,
    aggregation_type: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate daily time series data based on sample dates.

    Args:
        daily_data (pd.DataFrame): Daily time series with date index and 'evapotranspiration_[mm/d]' column.
        validation_dates (pd.DataFrame): DataFrame with 'date' column containing validation dates.
        aggregation_type (str): Type of aggregation to perform ('mean' or 'sum'). Defaults to 'mean'.

    Returns:
        pd.DataFrame: Aggregated data with sample dates and aggregated values.
    """
    daily_data.index = pd.to_datetime(daily_data.index)

    # Convert validation_dates to datetime and sort
    validation_dates["date"] = pd.to_datetime(validation_dates["date"])
    validation_dates = validation_dates.sort_values("date")

    # Create bins for grouping
    bins = validation_dates["date"].tolist() + [
        daily_data.index.max() + pd.Timedelta(days=1)
    ]

    # Cut the daily data into groups based on the bins
    daily_data["group"] = pd.cut(
        daily_data.index, bins=bins, labels=validation_dates["date"], right=False
    )

    # Group by the cut and calculate the aggregation
    if aggregation_type == "sum":
        aggregated_data = (
            daily_data.groupby("group")["evapotranspiration_[mm/d]"].sum().reset_index()
        )
        aggregated_data.columns = ["date", "sum_evapotranspiration_[mm/d]"]
    else:
        aggregated_data = (
            daily_data.groupby("group")["evapotranspiration_[mm/d]"]
            .mean()
            .reset_index()
        )
        aggregated_data.columns = ["date", "average_evapotranspiration_[mm/d]"]

    return aggregated_data
