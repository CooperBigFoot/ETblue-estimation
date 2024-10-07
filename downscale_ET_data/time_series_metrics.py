import numpy as np
import pandas as pd
from scipy import stats
from dtaidistance import dtw


def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) metric.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: R-squared value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return round(1 - (ss_res / ss_tot), 1)


def correlation_coefficient(y_true, y_pred):
    """
    Calculate the Pearson correlation coefficient.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: Correlation coefficient
    """
    return round(stats.pearsonr(y_true, y_pred)[0], 1)


def dynamic_time_warping(y_true, y_pred):
    """
    Calculate the Dynamic Time Warping distance.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: DTW distance
    """
    return round(dtw.distance(y_true, y_pred), 1)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: MAE value
    """
    return round(np.mean(np.abs(y_true - y_pred)), 1)


def root_mean_square_error(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: RMSE value
    """
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 1)


def normalized_root_mean_square_error(y_true, y_pred):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE).

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: NRMSE value
    """
    return round(
        root_mean_square_error(y_true, y_pred) / (np.max(y_true) - np.min(y_true)), 1
    )


def calculate_metrics(model_results, validation_data):
    """
    Calculate R-squared, Correlation Coefficient, Dynamic Time Warping, MAE, RMSE, MAPE, and NRMSE metrics.

    Args:
        model_results (pd.DataFrame): DataFrame with date index and model results
        validation_data (pd.DataFrame): DataFrame with date index and validation data

    Returns:
        dict: Dictionary containing the calculated metrics
    """
    # Ensure the indices match and sort by date
    common_dates = model_results.index.intersection(validation_data.index)
    model_values = model_results.loc[common_dates].sort_index().values.flatten()
    validation_values = validation_data.loc[common_dates].sort_index().values.flatten()

    r2 = r_squared(validation_values, model_values)
    corr = correlation_coefficient(validation_values, model_values)
    dtw_distance = dynamic_time_warping(validation_values, model_values)
    mae = mean_absolute_error(validation_values, model_values)
    rmse = root_mean_square_error(validation_values, model_values)
    nrmse = normalized_root_mean_square_error(validation_values, model_values)

    return {
        "R-squared": r2,
        "Correlation Coefficient": corr,
        "Dynamic Time Warping Distance": dtw_distance,
        "Mean Absolute Error": mae,
        "Root Mean Square Error": rmse,
        "Normalized Root Mean Square Error": nrmse,
    }
