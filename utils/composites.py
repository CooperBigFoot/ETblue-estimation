# File: /src/gee_processing/time_series_utils.py

import ee
from typing import List, Dict, Any, Optional


def harmonized_ts(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_intervals: List[List[ee.Date]],
    options: Optional[Dict[str, Any]] = None,
) -> ee.ImageCollection:
    """
    Generates a harmonized time series from a Sentinel-2 Image Collection.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_intervals (List[List[ee.Date]]): List of time intervals, each defined by a start and end ee.Date.
        options (Optional[Dict[str, Any]]): Optional parameters.

    Returns:
        ee.ImageCollection: A collection of aggregated images sorted by time.
    """
    options = options or {}
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    time_intervals = ee.List(time_intervals)

    def _stack_bands(time_interval, stack):
        outputs = aggregate_stack(
            masked_collection,
            band_list,
            time_interval,
            {"agg_type": agg_type, "band_name": band_name},
        )
        return ee.List(stack).add(ee.Image(outputs))

    stack = ee.List([])
    agg_stack = ee.List(time_intervals).iterate(_stack_bands, stack)

    return ee.ImageCollection(ee.List(agg_stack)).sort("system:time_start")


def aggregate_stack(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_interval: ee.List,
    options: Dict[str, Any],
) -> ee.Image:
    """
    Generates a temporally-aggregated image for a given time interval.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_interval (ee.List): A list containing start and end ee.Date objects for the interval.
        options (Dict[str, Any]): Optional parameters.

    Returns:
        ee.Image: An aggregated image for the specified time interval.
    """
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    time_interval = ee.List(time_interval)

    start_date = ee.Date(time_interval.get(0))
    end_date = ee.Date(time_interval.get(1))
    agg_interval_days = end_date.difference(start_date, "day")

    timestamp = {
        "system:time_start": start_date.advance(
            ee.Number(agg_interval_days.divide(2)).ceil(), "day"
        ).millis()
    }

    filtered_collection = masked_collection.filterDate(start_date, end_date).select(
        band_list
    )

    def create_empty_image():
        empty_image = ee.Image.constant(0).rename(band_list[0])
        for band in band_list[1:]:
            empty_image = empty_image.addBands(ee.Image.constant(0).rename(band))
        return empty_image.set(timestamp).float()

    def apply_reducer(reducer):
        return filtered_collection.reduce(reducer).rename(band_list).set(timestamp)

    if agg_type == "geomedian":
        reducer = ee.Reducer.geometricMedian(len(band_list))
    elif agg_type == "mean":
        reducer = ee.Reducer.mean()
    elif agg_type == "max":
        reducer = ee.Reducer.max()
    elif agg_type == "min":
        reducer = ee.Reducer.min()
    else:  # default to median
        reducer = ee.Reducer.median()

    return ee.Algorithms.If(
        filtered_collection.size().gt(0), apply_reducer(reducer), create_empty_image()
    )
