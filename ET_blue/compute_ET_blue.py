import ee


def compute_et_blue(et_total: ee.Image, et_green: ee.Image) -> ee.Image:
    """
    Compute ET blue by subtracting ET green from total ET.
    Apply a threshold to ET blue values.

    Args:
        et_total (ee.Image): Image containing total ET values.
        et_green (ee.Image): Image containing ET green values.
    Returns:
        ee.Image: Image containing ET blue values above the threshold.
    """

    date = et_total.get("system:time_start")

    et_blue = et_total.subtract(et_green).rename("ET_blue")

    return et_blue.set("system:time_start", date)


def compute_volumetric_et_blue(et_blue: ee.Image) -> ee.Image:
    """
    Convert ET blue from mm to cubic meters.

    Args:
        et_blue (ee.Image): Image containing ET blue values in mm.

    Returns:
        ee.Image: Image containing ET blue values in cubic meters.
    """

    date = et_blue.get("system:time_start")
    # Convert mm to m (divide by 1000) and multiply by pixel area
    return (
        et_blue.multiply(0.001)
        .multiply(ee.Image.pixelArea())
        .rename("ET_blue_m3")
        .set("system:time_start", date)
    )


def sum_et_blue_for_period(et_blue_collection: ee.ImageCollection) -> ee.Image:
    """
    Sum ET blue volumes over a specified period (e.g., growing season or year).

    Args:
        et_blue_collection (ee.ImageCollection): Collection of ET blue images in cubic meters.

    Returns:
        ee.Image: Image containing the sum of ET blue volumes for the period.
    """
    return et_blue_collection.sum().rename("ET_blue_m3_sum")


def compute_et_blue_per_ha_year(
    feature: ee.Feature, et_blue_sum: ee.Image
) -> ee.Feature:
    """
    Compute ET blue in m³/ha/year for a given feature.

    Args:
        feature (ee.Feature): The feature (field) to compute ET blue for.
        et_blue_sum (ee.Image): Image containing the sum of ET blue volumes for the year.

    Returns:
        ee.Feature: The input feature with an added 'ET_blue_m3_ha_year' property.
    """
    area = feature.geometry().area()
    et_blue_sum_feature = et_blue_sum.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=feature.geometry(), scale=30, maxPixels=1e9
    ).get("ET_blue_m3_sum")

    et_blue_m3_ha_year = ee.Number(et_blue_sum_feature).divide(area.divide(10000))
    return feature.set("ET_blue_m3_ha_year", et_blue_m3_ha_year)


def postprocess_et_blue(
    et_blue_image_present: ee.Image, et_blue_image_past: ee.Image, threshold: float
) -> ee.Image:
    """
    Postprocess ET blue images based on current and past values and a threshold.

    Args:
        et_blue_image_present (ee.Image): Current ET blue image.
        et_blue_image_past (ee.Image): Past ET blue image.
        threshold (float): Threshold value for ET blue.

    Returns:
        ee.Image: Postprocessed ET blue image.
    """
    date = et_blue_image_present.get("system:time_start")
    # Create a condition mask
    condition = et_blue_image_present.gte(threshold).And(
        et_blue_image_present.add(et_blue_image_past.min(0)).gt(0)
    )

    # Apply the condition: if true, return et_blue_image_present, otherwise return 0
    return ee.Image(
        et_blue_image_present.where(condition, 0)
        .rename("ET_blue")
        .set("system:time_start", date)
    )


# Example usage:
# et_blue = compute_et_blue(et_total_image, et_green_image)
# et_blue_vol = compute_volumetric_et_blue(et_blue)
# et_blue_sum = sum_et_blue_for_period(et_blue_collection)
# fields_with_et_blue = fields.map(lambda f: compute_et_blue_per_ha_year(f, et_blue_sum))
