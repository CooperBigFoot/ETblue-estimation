# File: /src/veg_period_extraction/data_loading.py

import ee
from typing import Dict, Any
from utils.s2_mask import load_image_collection, add_cloud_shadow_mask

# Initialize Earth Engine
ee.Initialize(project="thurgau-irrigation")


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
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
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
    ndvi_int = image.select("NDVI").multiply(10000).toInt16().rename("NDVI_int")
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
    date = ee.Date(image.get("system:time_start"))  # contains Unix time in milliseconds
    years = date.difference(
        ee.Date("1970-01-01"), "year"
    )  # difference in years from Unix time

    return image.addBands(ee.Image(years).rename("t").float()).addBands(
        ee.Image.constant(1)
    )


def get_s2_images(
    aoi: ee.Geometry, year: int, options: Dict[str, Any]
) -> ee.ImageCollection:
    """
    Get Sentinel-2 images for a specific area and year with custom options.

    Args:
        aoi (ee.Geometry): Area of interest.
        year (int): Year to filter images.
        options (Dict[str, Any]): Dictionary of options including 'cc_pix', 'min_m', and 'max_m'.

    Returns:
        ee.ImageCollection: Filtered and processed Sentinel-2 image collection.
    """
    cc_pix = options.get("cc_pix", 50)
    min_m = options.get("min_m", 1)
    max_m = options.get("max_m", 12)

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cc_pix))
        .map(lambda img: cloud_score_sentinel(img, aoi))
    )

    return s2_collection.select(["Blue", "Green", "Red", "NIR", "cloud", "NDVI"]).sort(
        "system:time_start"
    )


def cloud_score_sentinel(image: ee.Image, aoi: ee.Geometry) -> ee.Image:
    """
    Apply cloud scoring to Sentinel-2 image.

    Args:
        image (ee.Image): Input Sentinel-2 image.
        aoi (ee.Geometry): Area of interest for cloud pixel calculation.

    Returns:
        ee.Image: Cloud-scored image with additional properties.
    """
    # Get the date of the image
    image_date = ee.Date(image.get("system:time_start"))

    # Load the Sentinel-2 cloud probability image collection
    cloud_probability_collection = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(image_date.advance(-1, "hour"), image_date.advance(1, "hour"))
    )

    # Get the cloud probability image or create a default image if none is found
    cloud_probability_image = ee.Algorithms.If(
        cloud_probability_collection.size().gt(0),
        cloud_probability_collection.mosaic().select("probability"),
        ee.Image(0).rename("probability"),
    )

    # Calculate NDVI
    ndvi_image = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Create a cloud mask
    cloud_mask = (
        ee.Image(1)
        .where(
            image.select("QA60").lt(1024),
            ee.Image(1).where(image.select("B1").gt(0), 0),
        )
        .rename("cloud")
        .where(ndvi_image.gt(0.99), 1)
        .where(ee.Image(cloud_probability_image).select("probability").gt(50), 1)
    )

    # Calculate the mean cloud cover within the area of interest
    mean_cloud_cover = (
        cloud_mask.select("cloud")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13,
            tileScale=16,
        )
        .get("cloud")
    )

    # Return the cloud-scored image with additional properties
    return (
        image.rename(all_bands_s2)
        .addBands(ndvi_image)
        .updateMask(cloud_mask.Not())
        .addBands(cloud_mask.multiply(ee.Image(100)))
        .set(
            {
                "nodata_cover": mean_cloud_cover,
                "CLOUD_COVER": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            }
        )
    )


# Define constants
all_bands_s2 = [
    "Aerosols",
    "Blue",
    "Green",
    "Red",
    "Red Edge 1",
    "Red Edge 2",
    "Red Edge 3",
    "NIR",
    "Red Edge 4",
    "Water vapor",
    "Cirrus",
    "SWIR1",
    "SWIR2",
    "QA10",
    "QA20",
    "QA60",
    "MSK_CLASSI_OPAQUE",
    "MSK_CLASSI_CIRRUS",
    "MSK_CLASSI_SNOW_ICE",
]
