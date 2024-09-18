import ee
from typing import List, Dict, Any, Optional

# Initialize Earth Engine
ee.Initialize(project="thurgau-irrigation")

# Constants
DEFAULT_BAND_NAME = "NDVI"
DEFAULT_AGG_TYPE = "median"

REDUCERS = {
    "geomedian": ee.Reducer.geometricMedian,
    "mean": ee.Reducer.mean,
    "max": ee.Reducer.max,
    "min": ee.Reducer.min,
    "median": ee.Reducer.median,
}


def harmonized_ts(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_intervals: List[List[ee.Date]],
    options: Optional[Dict[str, Any]] = None,
) -> ee.ImageCollection:
    """
    Generates a harmonized time series from a Sentinel-2 Image Collection.

    Harmonized means that the generated temporal aggregates are equally spaced in time,
    based on the number of days specified by the time intervals.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_intervals (List[List[ee.Date]]): List of time intervals, each defined by a start and end ee.Date.
        options (Optional[Dict[str, Any]]): Optional parameters.
            - band_name (str): Name of the band for metadata. Defaults to 'NDVI'.
            - agg_type (str): Type of aggregation ('median', 'mean', 'geomedian', 'max', 'min'). Defaults to 'median'.

    Returns:
        ee.ImageCollection: A collection of aggregated images sorted by time.
    """
    if not isinstance(masked_collection, ee.ImageCollection):
        raise TypeError("masked_collection must be an ee.ImageCollection")

    options = ee.Dictionary(options or {})
    band_name = options.get("band_name", DEFAULT_BAND_NAME)
    agg_type = options.get("agg_type", DEFAULT_AGG_TYPE)

    def _stack_bands(time_interval: ee.List, stack: ee.List) -> ee.List:
        outputs = aggregate_stack(
            masked_collection,
            band_list,
            time_interval,
            ee.Dictionary({"agg_type": agg_type, "band_name": band_name}),
        )
        return ee.List(stack).add(ee.Image(outputs))

    initial_stack = ee.List([])
    agg_stack = ee.List(time_intervals).iterate(_stack_bands, initial_stack)

    return ee.ImageCollection(ee.List(agg_stack)).sort("system:time_start")


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
        options (ee.Dictionary): Optional parameters.
            - band_name (str): Name of the band for metadata. Defaults to 'NDVI'.
            - agg_type (str): Type of aggregation ('median', 'mean', 'geomedian', 'max', 'min'). Defaults to 'median'.

    Returns:
        ee.Image: An aggregated image for the specified time interval.
    """
    band_name = options.get("band_name", DEFAULT_BAND_NAME)
    agg_type = options.get("agg_type", DEFAULT_AGG_TYPE)

    time_interval = ee.List(time_interval)

    start_date = ee.Date(time_interval.get(0))
    end_date = ee.Date(time_interval.get(1))
    agg_interval_days = end_date.difference(start_date, "day")

    timestamp = {
        "system:time_start": start_date.advance(
            agg_interval_days.divide(2).ceil(), "day"
        ).millis()
    }

    def create_empty_image() -> ee.Image:
        return ee.Image.constant(0).rename(band_list).float().set(timestamp)

    filtered_collection = masked_collection.filterDate(start_date, end_date).select(
        band_list
    )

    reducer = REDUCERS.get(agg_type, ee.Reducer.median)
    if agg_type == "geomedian":
        reducer = reducer(len(band_list))

    agg_image = ee.Algorithms.If(
        filtered_collection.size().gt(0),
        filtered_collection.reduce(reducer()).rename(band_list).set(timestamp),
        create_empty_image(),
    )

    return ee.Image(agg_image)
