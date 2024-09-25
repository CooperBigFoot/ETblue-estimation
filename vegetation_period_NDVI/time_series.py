import ee
import math
from typing import List, Dict, Any
from utils.composites import harmonized_ts
from data_loading import (
    load_sentinel2_data,
    ndvi_band_to_int,
    ndvi_band_to_float,
    add_time_data,
)

# Initialize Earth Engine
ee.Initialize(project="thurgau-irrigation")


def extract_time_ranges(time_range: List[str], agg_interval: int) -> ee.List:
    """
    Extract time intervals for generating temporal composites from Sentinel collections.

    Args:
        time_range (List[str]): Start and end dates in 'YYYY-MM-DD' format.
        agg_interval (int): Number of days for each interval.

    Returns:
        ee.List: List of time intervals. Each interval is an ee.List with [start_date, end_date].
    """
    start_date = ee.Date(time_range[0])
    end_date = ee.Date(time_range[1])

    interval_no = (
        ee.Date(time_range[1])
        .difference(ee.Date(time_range[0]), "day")
        .divide(agg_interval)
        .round()
    )
    month_check = ee.Number(30).divide(agg_interval).ceil()
    rel_delta = (
        ee.Number(end_date.difference(start_date, "day"))
        .divide(ee.Number(30.5).multiply(interval_no))
        .ceil()
    )

    end_date = start_date.advance(
        start_date.advance(rel_delta, "month")
        .difference(start_date, "day")
        .divide(month_check),
        "day",
    )

    time_intervals = ee.List([ee.List([start_date, end_date])])

    def add_interval(x, previous):
        x = ee.Number(x)
        start_date1 = ee.Date(
            ee.List(ee.List(previous).reverse().get(0)).get(1)
        )  # end_date of last element
        end_date1 = start_date1.advance(
            start_date1.advance(rel_delta, "month")
            .difference(start_date1, "day")
            .divide(month_check),
            "day",
        )
        return ee.List(previous).add(ee.List([start_date1, end_date1]))

    time_intervals = ee.List(
        ee.List.sequence(2, interval_no).iterate(add_interval, time_intervals)
    )

    return time_intervals


OMEGA = 1.5  # Temporal frequency. Value from: https://doi.org/10.1016/j.rse.2018.12.026
MAX_HARMONIC_ORDER = 2  # Maximum order of harmonics


def add_harmonic_components(image: ee.Image) -> ee.Image:
    """Add harmonic component bands to the image."""
    for i in range(1, MAX_HARMONIC_ORDER + 1):
        time_radians = image.select("t").multiply(2 * i * OMEGA * math.pi)
        image = image.addBands(time_radians.cos().rename(f"cos{i}")).addBands(
            time_radians.sin().rename(f"sin{i}")
        )
    return image


def compute_harmonic_fit(
    vegetation_index: str,
    input_image_collection: ee.ImageCollection,
    parallel_scale: int,
) -> ee.ImageCollection:
    """
    Compute fitted values using harmonic regression on a vegetation index time series.
    The equation for the harmonic regression is:

        y(t) = b0 + sum(b2i*cos(2*pi*i*omega*t) + b2i+1*sin(2*pi*i*omega*t)) for i = 1 to n

    as in the paper: https://doi.org/10.1016/j.rse.2018.12.026

    Args:
        vegetation_index (str): Name of the vegetation index.
        input_image_collection (ee.ImageCollection): Input image collection.
        parallel_scale (int): Parallelization scale for the linear regression.

    Returns:
        ee.ImageCollection: Image collection with fitted values and RMSE.
    """
    harmonic_component_names = ["constant"] + [
        f"{trig}{i}"
        for i in range(1, MAX_HARMONIC_ORDER + 1)
        for trig in ["cos", "sin"]
    ]

    harmonic_component_names = ee.List(harmonic_component_names)

    harmonic_image_collection = input_image_collection.select(
        ee.List([ee.String(vegetation_index), "t", "constant"])
    ).map(add_harmonic_components)

    regression_input_bands = ee.List(harmonic_component_names).add(
        ee.String(vegetation_index)
    )

    regression_result = harmonic_image_collection.select(regression_input_bands).reduce(
        reducer=ee.Reducer.linearRegression(harmonic_component_names.length(), 1),
        parallelScale=parallel_scale,
    )

    regression_coefficients = (
        regression_result.select("coefficients")
        .arrayProject([0])
        .arrayFlatten([harmonic_component_names])
    )

    def compute_fitted_values_and_performance(image: ee.Image) -> ee.Image:
        """Compute fitted values and pwrformance metrics, and add them as new bands to the image.

        Args:
            image (ee.Image): Input image.

        Returns:
            ee.Image: Image with fitted values, RMSE and PBIAS."""
        fitted_values = (
            image.select(harmonic_component_names)
            .multiply(regression_coefficients)
            .reduce(ee.Reducer.sum())
            .rename("fitted")
        )

        rmse = (
            image.select(vegetation_index)
            .subtract(fitted_values)
            .pow(2)
            .sqrt()
            .rename("rmse")
        )

        return image.addBands(fitted_values).addBands(rmse)

    return harmonic_image_collection.map(compute_fitted_values_and_performance)


def calculate_phase_amplitude(regression_coefficients: ee.Image) -> ee.Image:
    """Calculate phase and amplitude from regression coefficients."""
    phase = ee.Image.constant(0)
    amplitude = ee.Image.constant(0)

    for i in range(1, MAX_HARMONIC_ORDER + 1):
        cos_band = f"cos{i}"
        sin_band = f"sin{i}"

        phase_i = (
            regression_coefficients.select(sin_band)
            .atan2(regression_coefficients.select(cos_band))
            .rename(f"phase{i}")
        )

        amplitude_i = (
            regression_coefficients.select(sin_band)
            .hypot(regression_coefficients.select(cos_band))
            .rename(f"amplitude{i}")
        )

        phase = phase.addBands(phase_i)
        amplitude = amplitude.addBands(amplitude_i)

    return phase.addBands(amplitude)


def get_harmonic_ts(
    year: int, aoi: ee.Geometry, time_intervals: ee.List
) -> Dict[str, Any]:
    """
    Generate a harmonized time series with harmonic regression for a given year and area.

    Args:
        year (int): The year for which to generate the time series.
        aoi (ee.Geometry): The area of interest.
        time_intervals (ee.List): List of time intervals for aggregation.

    Returns:
        Dict[str, Any]: Dictionary containing fitted data, regression coefficients, and phase/amplitude.
    """
    # Load Sentinel-2 data for the specified year and area
    yearly_sentinel_data = load_sentinel2_data(year, aoi)

    # # Convert NDVI to integer representation for cloud filtering
    yearly_sentinel_data = yearly_sentinel_data.map(ndvi_band_to_int)

    # Create harmonized time series
    harmonized_data = harmonized_ts(
        yearly_sentinel_data,
        ["NDVI_int", "NDVI"],
        time_intervals,
        {"agg_type": "geomedian"},
    ).map(lambda img: ndvi_band_to_float(ee.Image(img)))

    # Add time and constant bands
    harmonized_data = harmonized_data.map(add_time_data)

    # Interpolate missing data
    def interpolate_image(image: ee.Image) -> ee.Image:
        """
        Interpolate the image by averaging over a 30-day window centered on the current date.
        """
        current_date = ee.Date(image.get("system:time_start"))
        mean_filtered_image = harmonized_data.filterDate(
            current_date.advance(-15, "day"), current_date.advance(15, "day")
        ).mean()
        return mean_filtered_image.where(image, image).copyProperties(
            image, ["system:time_start"]
        )

    harmonized_data = ee.ImageCollection(harmonized_data.map(interpolate_image))

    # Compute harmonic fit
    fitted_data = compute_harmonic_fit("NDVI", harmonized_data, 2)

    # Extract regression coefficients from the first image
    regression_coefficients = fitted_data.first().select(
        ["constant"]
        + [
            f"{trig}{i}"
            for i in range(1, MAX_HARMONIC_ORDER + 1)
            for trig in ["cos", "sin"]
        ]
    )

    # Calculate phase and amplitude
    phase_amplitude = calculate_phase_amplitude(regression_coefficients)

    return {
        "fitted_data": fitted_data,
        "regression_coefficients": regression_coefficients,
        "phase_amplitude": phase_amplitude,
    }
