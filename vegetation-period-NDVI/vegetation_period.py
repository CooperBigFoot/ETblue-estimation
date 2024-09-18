import ee


def create_binary_ndvi_indicator(img: ee.Image, threshold: float) -> ee.Image:
    """
    Creates a binary NDVI indicator based on a threshold value. Values greater than the threshold are set to 1, and values less than or equal to the threshold are set to 0.

    Args:
        img (ee.Image): An image with an 'NDVI' band.
        threshold (float): The threshold value.

    Returns:
        ee.Image: A binary image with values of 0 or 1.
    """
    return (
        img.select("NDVI")
        .gt(threshold)
        .set("system:time_start", img.get("system:time_start"))
    )
