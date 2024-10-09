import ee


def compute_et_green(
    et_image: ee.Image, rainfed_reference: ee.FeatureCollection
) -> ee.Image:
    """
    Compute ET green based on the given ET image and rainfed reference areas.

    Args:
        et_image (ee.Image): An image containing ET values.
        rainfed_reference (ee.FeatureCollection): A feature collection of rainfed reference areas.

    Returns:
        ee.Image: An image with a single band 'ET_green' containing the computed ET green value.
    """
    projection = et_image.projection()
    scale = projection.nominalScale()
    time_start = et_image.get("system:time_start")

    # Compute the mean ET value over the rainfed reference areas
    mean_et = et_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=rainfed_reference.geometry(),
        scale=et_image.projection().nominalScale(),
        maxPixels=1e9,
    ).get("downscaled")

    # Check if the computed mean is valid, multiply by 100, and convert to integer
    et_green = ee.Image.constant(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(mean_et, None),
            ee.Number(0),
            ee.Number(mean_et)
            .multiply(100)
            .int(),  # Storing floats is expensive, so we multiply by 100 and convert to int
        )
    ).rename("ET_green")

    return et_green.setDefaultProjection(projection, None, scale).set(
        "system:time_start", time_start
    )


if __name__ == "__main__":
    # Example usage
    et_image = ee.Image("path/to/your/et/image")  # Replace with actual ET image path
    rainfed_areas = ee.FeatureCollection(
        "path/to/your/rainfed/areas"
    )  # Replace with actual rainfed areas path

    try:
        et_green_result = compute_et_green(et_image, rainfed_areas)
        print("ET green computation successful")

        # Example: Print the ET green value
        print(et_green_result.getInfo())

    except ee.EEException as e:
        print(f"An error occurred: {str(e)}")
