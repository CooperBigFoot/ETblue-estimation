import ee
from typing import List, Dict, Any, Optional

# Initialize the Earth Engine module.
ee.Initialize(project="thurgau-irrigation")


def harmonized_ts(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_intervals: ee.List,
    options: Optional[Dict[str, Any]] = None,
) -> ee.ImageCollection:
    """
    Generates a harmonized time series from a Sentinel-2 Image Collection.

    Harmonized means that the generated temporal aggregates are equally spaced in time,
    based on the number of days specified by the "agg_interval" argument.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_intervals (ee.List): List of time intervals, each defined by a start and end ee.Date.
        options (Dict[str, Any]): Optional parameters.
            - band_name (str): Name of the band for metadata. Defaults to 'NDVI'.
            - agg_type (str): Type of aggregation ('median', 'mean', 'geomedian', 'max', 'min'). Defaults to 'median'.

    Returns:
        ee.ImageCollection: A collection of aggregated images sorted by time.
    """
    options = options or {}
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    def _stack_bands(time_interval, stack):
        """
        Wrapper function for stacking the generated Sentinel-2 temporal aggregates.

        Args:
            time_interval (ee.List): A list containing start and end ee.Date objects for the interval.
            stack (ee.List): The current list of aggregated images.

        Returns:
            ee.List: Updated list with the new aggregated image added.
        """
        aggregated_image = aggregate_stack(
            masked_collection,
            band_list,
            ee.List(time_interval),
            {"agg_type": agg_type, "band_name": band_name},
        )
        return ee.List(stack).add(aggregated_image)

    # Initialize an empty list to hold the aggregated images.
    initial_stack = ee.List([])

    # Iterate over each time interval and accumulate the aggregated images.
    agg_stack = ee.List(time_intervals).iterate(_stack_bands, initial_stack)

    # Convert the list to an ImageCollection and sort by 'system:time_start'.
    return ee.ImageCollection(agg_stack).sort("system:time_start")


def aggregate_stack(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_interval: ee.List,
    options: Optional[Dict[str, Any]] = None,
) -> ee.Image:
    """
    Generates a temporally-aggregated image for a given time interval.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_interval (ee.List): A list containing start and end ee.Date objects for the interval.
        options (Dict[str, Any]): Optional parameters.
            - band_name (str): Name of the band for metadata. Defaults to 'NDVI'.
            - agg_type (str): Type of aggregation ('median', 'mean', 'geomedian', 'max', 'min'). Defaults to 'median'.

    Returns:
        ee.Image: An aggregated image for the specified time interval.
    """
    options = options or {}
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    start_date = ee.Date(time_interval.get(0))
    end_date = ee.Date(time_interval.get(1))
    agg_interval_days = end_date.difference(start_date, "day")

    # Set the center of the time interval as the 'system:time_start' date.
    timestamp = {
        "system:time_start": start_date.advance(
            agg_interval_days.divide(2).ceil(), "day"
        ).millis()
    }

    def create_empty_image(bands: List[str]) -> ee.Image:
        """
        Creates an empty image with the specified bands.

        Args:
            bands (List[str]): List of band names.

        Returns:
            ee.Image: An empty image with the specified bands.
        """
        empty_image = ee.Image().float()
        for band in bands:
            empty_image = empty_image.addBands(ee.Image().float().rename(band))
        return empty_image.rename(bands).set(timestamp)

    # Filter the collection for the current time interval.
    filtered_collection = masked_collection.filterDate(start_date, end_date).select(
        band_list
    )

    # Determine the aggregation reducer based on 'agg_type'.
    reducer = {
        "median": ee.Reducer.median(),
        "mean": ee.Reducer.mean(),
        "geomedian": ee.Reducer.geometricMedian(len(band_list)),
        "max": ee.Reducer.max(),
        "min": ee.Reducer.min(),
    }.get(agg_type, ee.Reducer.median())

    # Apply the aggregation if the filtered collection is not empty.
    agg_image = ee.Algorithms.If(
        filtered_collection.size().gt(0),
        filtered_collection.reduce(reducer).rename(band_list).set(timestamp),
        create_empty_image(band_list),
    )

    return ee.Image(agg_image)
