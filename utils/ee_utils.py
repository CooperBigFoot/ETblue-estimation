import ee
from typing import List


def set_negative_to_zero(image: ee.Image) -> ee.Image:
    """
    Set all negative values in an image to zero.

    Args:
        image (ee.Image): The input image.

    Returns:
        ee.Image: The image with all negative values set to zero.
    """
    return image.where(image.lt(0), 0)


def merge_collections(
    years: List[int], asset_name: str, special_char: str = None
) -> ee.ImageCollection:
    """
    Merge collections for multiple years.

    Args:
        years (list): List of years to process.
        asset_name (str): Name of the asset to merge.
        special_char (str): Special character to append to the asset name. Defaults to None.

    Returns:
        ee.ImageCollection: Merged collection for all years.
    """
    collections = [
        ee.ImageCollection(
            f"{asset_name}_{year}{f'_{special_char}' if special_char else ''}"
        )
        .sort("system:time_start")
        .map(set_negative_to_zero)
        for year in years
    ]

    # Merge all collections into one
    merged_collection = collections[0]
    for collection in collections[1:]:
        merged_collection = merged_collection.merge(collection)

    return merged_collection


def extract_pixel_values(
    image_collection: ee.ImageCollection,
    point: ee.Geometry.Point,
    band: str = "downscaled",
) -> ee.FeatureCollection:
    """
    Extract the pixel value of the specified band for each image in the collection
    at the specified point, with error handling for missing timestamps.

    Args:
        image_collection (ee.ImageCollection): The input image collection.
        point (ee.Geometry.Point): The point at which to extract values.
        band (str): The band to extract values from. Defaults to 'downscaled'.

    Returns:
        ee.FeatureCollection: A feature collection where each feature represents an image
                              and contains the pixel value of the "band" at the point.
    """

    def extract_value(image: ee.Image) -> ee.Feature:
        # Select the specified band
        image_band = image.select(band)

        # Get the scale of the specified band
        scale = image_band.projection().nominalScale()

        # Extract the pixel value at the point
        pixel_value = image_band.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale,
            bestEffort=True,
        ).get(band)

        # Retrieve the image acquisition time
        time_start = image.get("system:time_start")

        # Handle potential null time_start
        formatted_date = ee.Algorithms.If(
            ee.Algorithms.IsEqual(time_start, None),
            None,
            ee.Date(time_start).format("YYYY-MM-dd"),
        )

        return ee.Feature(
            None,
            {
                "pixel_value": pixel_value,
                "date": formatted_date,
                "system:time_start": time_start,
            },
        )

    # Map the extraction function over the image collection
    return ee.FeatureCollection(image_collection.map(extract_value))


def aggregate_to_monthly(
    collection: ee.ImageCollection, bands: List[str] = ["downscaled"]
) -> ee.ImageCollection:
    """
    Aggregate an image collection to monthly images, weighted by the number of days each image represents.

    Args:
        collection (ee.ImageCollection): Input collection.
        bands (List[str]): List of band names to aggregate. Defaults to ["downscaled"].

    Returns:
        ee.ImageCollection: Monthly aggregated image collection.
    """

    def aggregate_month(year, month, images):
        images = ee.List(images)
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")
        days_in_month = end_date.difference(start_date, "day")

        def weight_image(i):
            i = ee.Number(i)
            image = ee.Image(images.get(i))
            next_image = ee.Image(images.get(i.add(1)))
            date = ee.Date(image.get("system:time_start"))
            next_date = ee.Date(
                ee.Algorithms.If(
                    i.eq(images.size().subtract(1)),
                    end_date,
                    next_image.get("system:time_start"),
                )
            )
            weight = next_date.difference(date, "day")
            # Cast the selected bands to a consistent float type
            return (
                image.select(bands)
                .cast(
                    ee.Dictionary.fromLists(bands, ee.List.repeat("float", len(bands)))
                )
                .multiply(weight)
            )

        weighted_sum = ee.ImageCollection.fromImages(
            ee.List.sequence(0, images.size().subtract(1)).map(weight_image)
        ).sum()

        return weighted_sum.set(
            {"system:time_start": start_date.millis(), "year": year, "month": month}
        )

    # Get unique year-month combinations
    dates = collection.aggregate_array("system:time_start")
    unique_year_months = dates.map(lambda d: ee.Date(d).format("YYYY-MM")).distinct()

    def process_year_month(ym):
        ym = ee.String(ym)
        year = ee.Number.parse(ym.slice(0, 4))
        month = ee.Number.parse(ym.slice(5, 7))
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")

        monthly_images = collection.filterDate(start_date, end_date)
        return aggregate_month(
            year, month, monthly_images.toList(monthly_images.size())
        )

    aggregated = ee.ImageCollection.fromImages(
        unique_year_months.map(process_year_month)
    )

    projection = collection.first().projection()
    scale = projection.nominalScale()

    # Ensure consistent float type for the entire collection
    return aggregated.map(
        lambda img: img.cast(
            ee.Dictionary.fromLists(bands, ee.List.repeat("float", len(bands)))
        ).setDefaultProjection(projection, None, scale)
    ).sort("system:time_start")


def back_to_float(image: ee.Image, scale: int) -> ee.Image:
    """
    Convert an image to float and divide by the scale

    Args:
        image: The image to convert
        scale: The scale to divide by

    Returns:
        The image converted to float and divided by the scale
    """
    date = image.get("system:time_start")
    return image.toFloat().divide(scale).set("system:time_start", date)


def back_to_int(image: ee.Image, scale: int) -> ee.Image:
    """
    Convert an image to int and multiply by the scale

    Args:
        image: The image to convert
        scale: The scale to multiply by

    Returns:
        The image converted to int and multiplied by the scale
    """
    date = image.get("system:time_start")
    return image.multiply(scale).toInt().set("system:time_start", date)


def export_image_to_asset(
    image: ee.Image,
    asset_id: str,
    task_name: str,
    year: int,
    aoi: ee.Geometry,
    max_pixels: int = 1e13,
) -> ee.batch.Task:
    """
    Export an image to an Earth Engine asset.
    """
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=task_name,
        assetId=asset_id,
        region=aoi,
        scale=10,
        maxPixels=max_pixels,
    )
    print(f"Exporting {task_name} for {year} to {asset_id}")
    task.start()
    return task


def print_value_ranges(
    collection: ee.ImageCollection, band_name: str = "ET_blue"
) -> None:
    """
    Print the minimum and maximum values for each image in the collection.

    Args:
        collection (ee.ImageCollection): Collection of images to analyze
        band_name (str): Name of the band to analyze
    """

    def get_minmax(image):
        stats = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e9,
        )
        return image.set(
            {"min": stats.get(f"{band_name}_min"), "max": stats.get(f"{band_name}_max")}
        )

    # Map the minmax computation over the collection
    collection_with_stats = collection.map(get_minmax)

    # Get the stats as lists
    stats = (
        collection_with_stats.aggregate_array("min")
        .zip(collection_with_stats.aggregate_array("max"))
        .getInfo()
    )

    # Print results
    for i, (min_val, max_val) in enumerate(stats):
        print(f"Image {i + 1}: Min = {min_val:.2f}, Max = {max_val:.2f}")
