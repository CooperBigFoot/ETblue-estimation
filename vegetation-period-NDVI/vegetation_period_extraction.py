# File: /src/veg_period_extraction/vegetation_period_extractor.py

import ee
from typing import List, Dict, Any
from utils.composites import harmonized_ts
from time_series import get_harmonic_ts


def create_binary_ndvi_indicator(img: ee.Image, threshold: float) -> ee.Image:
    return (
        img.select("NDVI")
        .gt(threshold)
        .rename("vegetation")
        .set("system:time_start", img.get("system:time_start"))
    )


def create_binary_mask(
    ndvi_collection: ee.ImageCollection, threshold: float
) -> ee.ImageCollection:
    return ndvi_collection.map(lambda img: create_binary_ndvi_indicator(img, threshold))


def find_vegetation_periods(
    binary_mask: ee.ImageCollection,
    time_intervals: ee.List,
    agg_interval: int,
    reverse: bool = False,
) -> ee.Image:
    def check_interval(x: ee.Number, prev: ee.Image) -> ee.Image:
        x = ee.Number(x)
        interval = ee.List(time_intervals.get(x))
        start, end = ee.Date(interval.get(0)), ee.Date(interval.get(1))

        consecutive_veg = (
            binary_mask.filterDate(start, end.advance(agg_interval, "day")).sum().gte(2)
        )

        return ee.Image(x).updateMask(consecutive_veg).unmask(prev)

    sequence = (
        ee.List.sequence(time_intervals.length().subtract(1), 0, -1)
        if reverse
        else ee.List.sequence(0, time_intervals.length().subtract(1))
    )

    return sequence.iterate(check_interval, ee.Image.constant(-1))


def find_end_first_period(
    binary_mask: ee.ImageCollection,
    first_period: ee.Image,
    last_period: ee.Image,
    time_intervals: ee.List,
) -> ee.Image:
    def check_interval(m: ee.Number, prev: ee.Image) -> ee.Image:
        m = ee.Number(m)
        image1 = binary_mask.toList(99).get(m.subtract(1))
        image2 = binary_mask.toList(99).get(m)
        image3 = binary_mask.toList(99).get(m.add(1))

        condition = (
            ee.Image(image1)
            .add(ee.Image(image2))
            .eq(2)
            .And(ee.Image(image2).add(ee.Image(image3)).eq(1))
            .And(first_period.lte(m))
            .And(last_period.gte(m))
        )

        return ee.Image(m).updateMask(condition).unmask(prev)

    return ee.List.sequence(1, binary_mask.size().subtract(2)).iterate(
        check_interval, ee.Image.constant(-1)
    )


def detect_double_cropping(
    binary_mask: ee.ImageCollection,
    first_period: ee.Image,
    last_period: ee.Image,
    time_intervals: ee.List,
    agg_interval: int,
    ndvi_low_th: float,
) -> ee.Image:
    binary_mask_low = create_binary_mask(binary_mask, ndvi_low_th)

    def check_interval(x: ee.Number, prev: ee.Image) -> ee.Image:
        x = ee.Number(x)
        interval = ee.List(time_intervals.get(x))
        start, end = ee.Date(interval.get(0)), ee.Date(interval.get(1))

        no_veg_period = (
            binary_mask_low.filterDate(start, end.advance(agg_interval, "day"))
            .sum()
            .eq(0)
        )

        is_between_crops = first_period.add(ee.Image(1)).lte(x).And(last_period.gte(x))

        return ee.Image(x).updateMask(no_veg_period.And(is_between_crops)).unmask(prev)

    result = ee.List.sequence(0, time_intervals.length().subtract(1)).iterate(
        check_interval, ee.Image.constant(-1)
    )

    return ee.Image(result).gt(-1)


def convert_to_months(period_image: ee.Image, time_intervals: ee.List) -> ee.Image:
    start_month = ee.Date(ee.List(time_intervals.get(0)).get(0)).get("month")
    return period_image.divide(2).floor().add(ee.Number(start_month)).mod(12).add(1)


def extract_vegetation_periods(
    ndvi_collection: ee.ImageCollection,
    time_intervals: ee.List,
    ndvi_threshold: float,
    ndvi_low_threshold: float,
    agg_interval: int,
) -> Dict[str, ee.Image]:
    binary_mask = create_binary_mask(ndvi_collection, ndvi_threshold)

    first_period = ee.Image(
        find_vegetation_periods(binary_mask, time_intervals, agg_interval)
    )
    last_period = ee.Image(
        find_vegetation_periods(binary_mask, time_intervals, agg_interval, reverse=True)
    )

    end_first = find_end_first_period(
        binary_mask, first_period, last_period, time_intervals
    )

    end_first = ee.Image(end_first)
    end_first0 = last_period.where(end_first.neq(last_period), end_first)
    end_first = end_first.updateMask(end_first.neq(last_period))

    second_start = ee.Image(
        find_vegetation_periods(
            binary_mask,
            time_intervals,
            agg_interval,
            lambda img: img.updateMask(end_first0.lte(img.subtract(1))),
        )
    )

    double_cropping = detect_double_cropping(
        ndvi_collection,
        first_period,
        last_period,
        time_intervals,
        agg_interval,
        ndvi_low_threshold,
    )

    last_period = last_period.where(end_first.gt(0).And(double_cropping), end_first)

    result = {
        "first": convert_to_months(first_period, time_intervals),
        "last": convert_to_months(last_period, time_intervals),
        "double": double_cropping,
        "first2": convert_to_months(second_start, time_intervals),
        "last2": convert_to_months(last_period, time_intervals),
    }

    # Ensure the last month of the first crop is not equal to the first month of the second crop
    result["first2"] = result["first2"].where(
        result["last"].eq(result["first2"]), result["first2"].add(ee.Image(1))
    )

    # Convert to calendar year
    for key in ["first", "last", "first2", "last2"]:
        result[key] = result[key].where(result[key].gt(12), result[key].subtract(12))

    return result


def get_vegetation_periods(
    year: int, aoi: ee.Geometry, time_intervals: List[Dict[str, str]]
) -> Dict[str, ee.Image]:
    ndvi_collection = get_harmonic_ts(year, aoi, time_intervals)

    return extract_vegetation_periods(
        ndvi_collection,
        ee.List(time_intervals),
        ndvi_threshold=0.35,
        ndvi_low_threshold=0.3,
        agg_interval=15,
    )
