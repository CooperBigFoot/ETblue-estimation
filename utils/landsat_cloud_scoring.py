"""
Original code from: users/hydrosolutions/public_functions:SedimentBalance_T1L2
"""

import ee
from typing import List


def create_qa_mask(image: ee.Image) -> ee.Image:
    """Creates a QA mask from the 'QA_PIXEL' band.

    Args:
        image (ee.Image): The input Landsat image.

    Returns:
        ee.Image: The QA mask image.
    """
    qa_pixel = image.select("QA_PIXEL")
    qa_mask = qa_pixel.bitwiseAnd(31).eq(0)  # 31 is binary '11111'
    return qa_mask.rename("mask")


def create_cloud_mask(qa_mask: ee.Image) -> ee.Image:
    """Creates a cloud mask based on the QA mask.

    Args:
        qa_mask (ee.Image): The QA mask image.

    Returns:
        ee.Image: The cloud mask image.
    """
    cloud_mask = ee.Image(100).rename("cloud").where(qa_mask.eq(1), 0)
    return cloud_mask


def apply_scaling_factors(
    image: ee.Image, optical_band_pattern: str, thermal_band_names: List[str]
) -> ee.Image:
    """Applies scaling factors to optical and thermal bands.

    Args:
        image (ee.Image): The input Landsat image.
        optical_band_pattern (str): Pattern to select optical bands (e.g., 'SR_B.').
        thermal_band_names (List[str]): List of thermal band names.

    Returns:
        ee.Image: Image with scaled optical and thermal bands.
    """
    optical_bands = image.select(optical_band_pattern).multiply(0.0000275).add(-0.2)
    thermal_bands = image.select(thermal_band_names).multiply(0.00341802).add(149.0)
    return optical_bands.addBands(thermal_bands)


def calculate_cloud_cover(
    cloud_mask: ee.Image, area_of_interest: ee.Geometry
) -> ee.Number:
    """Calculates the cloud cover percentage over the area of interest.

    Args:
        cloud_mask (ee.Image): The cloud mask image.
        area_of_interest (ee.Geometry): The area of interest.

    Returns:
        ee.Number: The cloud cover percentage.
    """
    cloud_cover = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=area_of_interest, scale=100, tileScale=2
    ).get("cloud")
    return ee.Number(cloud_cover)


def cloudscore_landsat8(image: ee.Image, area_of_interest: ee.Geometry) -> ee.Image:
    """Applies cloud scoring to Landsat 8 imagery.

    Args:
        image (ee.Image): Input Landsat 8 image.
        area_of_interest (ee.Geometry): Area of interest for cloud cover calculation.

    Returns:
        ee.Image: Processed image with cloud mask and additional properties.
    """
    # Create masks
    qa_mask = create_qa_mask(image)
    cloud_mask = create_cloud_mask(qa_mask)
    cloud_mask_with_qa = cloud_mask.updateMask(image.select("QA_PIXEL").neq(0))

    # Apply scaling factors
    scaled_bands = apply_scaling_factors(
        image, optical_band_pattern="SR_B.", thermal_band_names=["ST_B10"]
    )

    # Calculate cloud cover statistics
    nodata_cover = calculate_cloud_cover(cloud_mask, area_of_interest)
    cloud_cover = calculate_cloud_cover(cloud_mask_with_qa, area_of_interest)

    # Prepare and return the final image
    final_image = (
        image.addBands(scaled_bands, overwrite=True)
        .updateMask(cloud_mask.lt(100))
        .addBands([cloud_mask, qa_mask])
        .set(
            {
                "nodata_cover": nodata_cover,
                "cloud_cover": cloud_cover,
                "SENSING_TIME": ee.Date(image.get("system:time_start")).format(
                    "YYYY-MM-dd"
                ),
                "SATELLITE": "LANDSAT_8",
            }
        )
    )

    return final_image


def cloudscore_landsat7(image: ee.Image, area_of_interest: ee.Geometry) -> ee.Image:
    """Applies cloud scoring to Landsat 7 imagery.

    Args:
        image (ee.Image): Input Landsat 7 image.
        area_of_interest (ee.Geometry): Area of interest for cloud cover calculation.

    Returns:
        ee.Image: Processed image with cloud mask and additional properties.
    """
    # Create masks
    qa_mask = create_qa_mask(image)
    cloud_mask = create_cloud_mask(qa_mask)
    cloud_mask_with_qa = cloud_mask.updateMask(image.select("QA_PIXEL").neq(0))

    # Apply scaling factors
    scaled_bands = apply_scaling_factors(
        image, optical_band_pattern="SR_B.", thermal_band_names=["ST_B6"]
    )

    # Calculate cloud cover statistics
    nodata_cover = calculate_cloud_cover(cloud_mask, area_of_interest)
    cloud_cover = calculate_cloud_cover(cloud_mask_with_qa, area_of_interest)

    # Prepare and return the final image
    final_image = (
        image.addBands(scaled_bands, overwrite=True)
        .updateMask(cloud_mask.lt(100))
        .addBands([cloud_mask, qa_mask])
        .set(
            {
                "nodata_cover": nodata_cover,
                "cloud_cover": cloud_cover,
                "SENSING_TIME": ee.Date(image.get("system:time_start")).format(
                    "YYYY-MM-dd"
                ),
                "SATELLITE": "LANDSAT_7",
            }
        )
    )

    return final_image
