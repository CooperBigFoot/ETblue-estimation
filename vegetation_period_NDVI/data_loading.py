# File: /src/veg_period_extraction/data_loading.py

import ee
from typing import Dict, Any
from utils.s2_mask import load_image_collection, add_cloud_shadow_mask


def load_sentinel2_data(year: int, aoi: ee.Geometry) -> ee.ImageCollection:
    """
    Load Sentinel-2 data for a given year and area of interest, applying cloud and shadow masks.

    Args:
        year (int): The year for which to load the data.
        aoi (ee.Geometry): The area of interest.

    Returns:
        ee.ImageCollection: The processed Sentinel-2 image collection.
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)

    s2_filtered = load_image_collection(
        "COPERNICUS/S2_HARMONIZED", {"start": start_date, "end": end_date}, aoi
    ).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))

    not_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(0)

    # Apply cloud and shadow masks
    s2_masked = s2_filtered.map(lambda img: add_cloud_shadow_mask(not_water)(img))

    return s2_masked.map(add_variables).map(add_time_data).set("sensor_id", 0)


def add_variables(image: ee.Image) -> ee.Image:
    """
    Add NDVI and LSWI bands to the image, and apply cloud masking.

    Args:
        image (ee.Image): Input Sentinel-2 image.

    Returns:
        ee.Image: Image with added bands and cloud mask applied.
    """
    ndvi = (
        image.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat() # clamp
    )
    lswi = image.normalizedDifference(["B8", "B11"]).rename(
        "LSWI"
    )  # Land Surface Water Index

    # Create cloud mask
    cloud_mask = (
        ee.Image(1)
        .where(
            image.select("QA60").lt(1024),
            ee.Image(1).where(image.select("B1").gt(0), 0),
        )
        .rename("cloud")
        .where(ndvi.gt(0.99), 1)
    )

    return (
        image.addBands(ndvi)
        .addBands(lswi)
        .updateMask(cloud_mask.Not())
        .addBands(cloud_mask.multiply(ee.Image(100)))
        .set({"CLOUD_COVER": image.get("CLOUDY_PIXEL_PERCENTAGE")})
    )


def ndvi_band_to_int(image: ee.Image) -> ee.Image:
    """
    Convert the NDVI band of the image to an integer representation.

    Args:
        image (ee.Image): Input image with NDVI band.

    Returns:
        ee.Image: Image with NDVI band converted to integer representation.
    """

    ndvi_int = image.select("NDVI").multiply(10000).toInt().rename("NDVI_int")
    return image.addBands(ndvi_int)


def ndvi_band_to_float(image: ee.Image) -> ee.Image:
    """
    Convert the NDVI band of the image from integer to float representation.

    Args:
        image (ee.Image): Input image with NDVI band in integer representation.

    Returns:
        ee.Image: Image with NDVI band converted to float representation.
    """

    ndvi_float = image.select("NDVI_int").toFloat().divide(10000).rename("NDVI")
    return image.addBands(ndvi_float, overwrite=True)


def add_time_data(image: ee.Image) -> ee.Image:
    """
    Add time-related bands to the image.

    Args:
        image (ee.Image): Input image.

    Returns:
        ee.Image: Image with added time-related bands.
    """
    date = ee.Date(image.get("system:time_start"))

    years = date.difference(ee.Date("1970-01-01"), "year")

    time_band = ee.Image(years).rename("t").float()
    constant_band = ee.Image.constant(1)

    return image.addBands(time_band).addBands(constant_band)