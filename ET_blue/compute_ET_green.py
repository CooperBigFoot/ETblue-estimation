import ee


def compute_et_green_by_grid(
    et_image: ee.Image,
    rainfed_reference: ee.FeatureCollection,
    region: ee.Geometry,
    grid_size: float = 10000,  # Default grid size in meters
) -> ee.Image:
    """
    Compute ET green based on the given ET image and rainfed reference areas using a grid.

    Args:
        et_image (ee.Image): An image containing ET values.
        rainfed_reference (ee.FeatureCollection): A feature collection of rainfed reference areas.
        region (ee.Geometry): The region to create the grid over.
        grid_size (float): Size of each grid cell in meters.

    Returns:
        ee.Image: An image with a single band 'ET_green' containing the computed ET green values for each grid cell.
    """
    projection = et_image.projection()
    scale = projection.nominalScale()
    time_start = et_image.get("system:time_start")

    # Create a grid
    grid = ee.FeatureCollection(region.coveringGrid(projection, grid_size))

    # Add a numeric property to rainfed_reference
    rainfed_reference = rainfed_reference.map(lambda f: f.set("dummy", 1))

    # Mask the ET image with rainfed reference areas
    masked_et = et_image.updateMask(
        rainfed_reference.reduceToImage(["dummy"], ee.Reducer.first()).mask()
    )

    # Compute the overall mean ET value (fallback for grid cells without rainfed areas)
    overall_mean_et = ee.Number(
        masked_et.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        ).get("downscaled")
    )

    # Compute mean ET values for each grid cell
    def compute_grid_cell_mean(feature):
        cell_mean = masked_et.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=scale,
            maxPixels=1e13,
        ).get("downscaled")

        # Use overall mean if grid cell has no valid ET values
        mean_et = ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(cell_mean, None), overall_mean_et, cell_mean
            )
        )

        return feature.set("mean_et", mean_et)

    grid_with_mean = grid.map(compute_grid_cell_mean)

    # Create an image with ET green values for each grid cell
    et_green = (
        grid_with_mean.reduceToImage(["mean_et"], ee.Reducer.first())
        .multiply(100)
        .int()
        .rename("ET_green")
    )

    return et_green.setDefaultProjection(projection, None, scale).set(
        "system:time_start", time_start
    )


def calculate_band_std_dev(
    image: ee.Image,
    band_name: str,
    region: ee.Geometry = None,
    scale: float = None,
    max_pixels: int = 1e9,
) -> float:
    """
    Calculate the standard deviation of values in a specified band of an Earth Engine image.

    Args:
        image (ee.Image): The input Earth Engine image.
        band_name (str): The name of the band to analyze.
        region (ee.Geometry, optional): The region over which to calculate the standard deviation.
            If None, the image bounds will be used.
        scale (float, optional): The scale in meters of the projection to work in.
            If None, the native scale of the image will be used.
        max_pixels (int, optional): The maximum number of pixels to sample.
            Default is 1e9 (1 billion pixels).

    Returns:
        float: The standard deviation of the values in the specified band.

    Raises:
        ee.EEException: If the specified band is not found in the image or if the computation fails.
    """
    # Select the specified band
    single_band_image = image.select(band_name)

    # If no region is specified, use the image bounds
    if region is None:
        region = single_band_image.geometry()

    # Ensure the region is not null or empty
    region = ee.Algorithms.If(
        ee.Algorithms.IsEqual(region, None), single_band_image.geometry(), region
    )

    # If no scale is specified, use the nominal scale of the image
    if scale is None:
        scale = single_band_image.projection().nominalScale()

    try:
        # Calculate standard deviation using reduceRegion
        std_dev_dict = single_band_image.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=region,
            scale=scale,
            maxPixels=max_pixels,
        )

        # Extract the standard deviation value
        std_dev = std_dev_dict.get(band_name)

        return ee.Number(std_dev).getInfo()

    except ee.EEException as e:
        print(f"Error calculating standard deviation: {str(e)}")
        raise
