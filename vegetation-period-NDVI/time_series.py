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


# TODO: check return types. The nested list should be ee.List but it is a ee.ComputedObject.. why?
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


def get_harmonic_ts(
    year: int, aoi: ee.Geometry, time_intervals: ee.List
) -> ee.ImageCollection:
    """
    Generate a harmonized time series with harmonic regression for a given year and area.

    Args:
        year (int): The year for which to generate the time series.
        aoi (ee.Geometry): The area of interest.
        time_intervals (ee.List): List of time intervals for aggregation.

    Returns:
        ee.ImageCollection: Harmonized time series with fitted values.
    """
    yearly_sentinel_data = load_sentinel2_data(year, aoi)

    cloud_filtered_data = yearly_sentinel_data.map(ndvi_band_to_int)

    harmonized_data = harmonized_ts(
        cloud_filtered_data,
        ["NDVI_int"],
        time_intervals,
        {"agg_type": "geomedian"},
    ).map(lambda img: ndvi_band_to_float(ee.Image(img)))

    # Add 't' and 'constant' bands after harmonization
    harmonized_data = harmonized_data.map(add_time_data)

    def replace_by_empty(
        harmonized_collection: ee.ImageCollection,
        cloud_filtered_collection: ee.ImageCollection,
        time_range: ee.List,
    ) -> ee.Image:
        """
        Replace the harmonized image with an empty image if no cloud-filtered images are found in the time range.
        """
        time_range = ee.List(time_range)
        start_date = ee.Date(time_range.get(0))
        end_date = ee.Date(time_range.get(1))
        nb_imgs = cloud_filtered_collection.filterDate(start_date, end_date).size()
        img = harmonized_collection.filterDate(start_date, end_date).first()

        def create_empty_image(str_name, prev):
            return ee.Image(prev).addBands(
                img.select(ee.String(str_name)).updateMask(ee.Image().mask())
            )

        empty_img = (
            ee.Image(img.bandNames().iterate(create_empty_image, ee.Image()))
            .select(img.bandNames())
            .copyProperties(img, ["system:time_start"])
        )

        return ee.Algorithms.If(nb_imgs.gt(0), img, empty_img)

    harmonized_data = ee.ImageCollection(
        ee.List(time_intervals).map(
            lambda list_item: replace_by_empty(
                harmonized_data, cloud_filtered_data, ee.List(list_item)
            )
        )
    )

    names = harmonized_data.first().bandNames()

    def interpolate_image(image: ee.Image) -> ee.Image:
        """
        Interpolate the image by averaging over a 30-day window centered on the current date.
        """
        current_date = ee.Date(image.get("system:time_start"))
        mean_filtered_image = harmonized_data.filterDate(
            current_date.advance(-15, "day"), current_date.advance(15, "day")
        ).mean()
        return (
            mean_filtered_image.where(image, image)
            .rename(names)
            .copyProperties(image, ["system:time_start"])
        )

    harmonized_data = ee.ImageCollection(harmonized_data.map(interpolate_image))

    return compute_fitted_values("NDVI", harmonized_data, 2)


def compute_fitted_values(
    vegetation_index: str,
    sentinel_image_collection: ee.ImageCollection,
    parallel_scale: int,
) -> ee.ImageCollection:
    """
    Compute fitted values using harmonic regression.

    Args:
        vegetation_index (str): Vegetation index to use (e.g., "NDVI").
        sentinel_image_collection (ee.ImageCollection): Input sensor data.
        parallel_scale (int): Parallel scale for computation.

    Returns:
        ee.ImageCollection: Image collection with fitted values.
    """
    harmonic_terms = ee.List(["constant", "t", "cos", "sin", "cos2", "sin2"])
    max_order = 2
    omega = 1

    def add_harmonic_terms(image: ee.Image) -> ee.Image:
        """
        Add harmonic terms to the image.
        """

        time_radians = image.select("t").multiply(2 * omega * math.pi)
        time_radians_2 = image.select("t").multiply(4 * omega * math.pi)

        return (
            image.addBands(time_radians.cos().rename("cos"))
            .addBands(time_radians.sin().rename("sin"))
            .addBands(time_radians_2.cos().rename("cos2"))
            .addBands(time_radians_2.sin().rename("sin2"))
        )

    harmonic_sentinel_collection = sentinel_image_collection.select(
        ee.List([ee.String(vegetation_index), "t", "constant"])
    ).map(add_harmonic_terms)

    harmonic_regression_result = harmonic_sentinel_collection.select(
        harmonic_terms.slice(0, ee.Number(max_order).multiply(2).add(1)).add(
            ee.String(vegetation_index)
        )
    ).reduce(
        ee.Reducer.linearRegression(
            harmonic_terms.slice(0, ee.Number(max_order).multiply(2).add(1)).length(),
            1,
        ),
        parallel_scale,
    )

    regression_coefficients = (
        harmonic_regression_result.select("coefficients")
        .arrayProject([0])
        .arrayFlatten(
            [harmonic_terms.slice(0, ee.Number(max_order).multiply(2).add(1))]
        )
    )

    def add_fitted_values(image: ee.Image) -> ee.Image:
        """
        Add fitted values to the image using harmonic regression coefficients.
        """
        fitted = (
            image.select(
                harmonic_terms.slice(0, ee.Number(max_order).multiply(2).add(1))
            )
            .multiply(regression_coefficients)
            .reduce(ee.Reducer.sum())
            .rename("fitted")
        )
        return image.addBands(fitted)

    return harmonic_sentinel_collection.map(add_fitted_values)
