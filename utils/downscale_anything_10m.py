import ee


def compute_residuals(original_image: ee.Image, modeled_image: ee.Image) -> ee.Image:
    """
    Computes the residuals between the original LST and the modeled LST.

    Args:
        original_image (ee.Image): Original Landsat 8 LST image.
        modeled_image (ee.Image): Modeled LST based on regression.

    Returns:
        ee.Image: Residuals image.
    """
    return original_image.subtract(modeled_image).rename("residuals")


def apply_gaussian_smoothing(image: ee.Image, radius: float = 1.5) -> ee.Image:
    """
    Applies Gaussian smoothing to an image.

    Args:
        image (ee.Image): Input image to smooth.
        radius (float): Radius of the Gaussian kernel in pixels.

    Returns:
        ee.Image: Smoothed image.
    """
    gaussian_kernel = ee.Kernel.gaussian(radius=radius, units="pixels")
    return image.resample("bicubic").convolve(gaussian_kernel)


def perform_regression(
    independent_vars: ee.Image, dependent_var: ee.Image, image_resolution: int
) -> ee.Dictionary:
    """
    Performs linear regression using independent variables to predict the dependent variable.

    Args:
        independent_vars (ee.Image): Image containing bands of independent variables. It should have three bands: NDVI, NDBI, and NDWI.
        dependent_var (ee.Image): Single-band image of the dependent variable.
        image_resolution (int): Resolution of the input images in meters.
    Returns:
        ee.Dictionary: The result of the linear regression.
    """

    # Make sure the band order is as follows: NDVI, NDBI, NDWI
    try:
        independent_vars = independent_vars.select(["NDVI", "NDBI", "NDWI"])
    except ee.EEException:
        raise ValueError(
            "The independent_vars image should have the following bands: NDVI, NDBI, NDWI."
        )

    all_vars = independent_vars.addBands(dependent_var)

    independent_names = independent_vars.bandNames()

    # Perform the linear regression
    regression_result = all_vars.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=independent_names.length(), numY=1),
        scale=image_resolution,
        maxPixels=1e13,
    )

    return regression_result


def extract_coefficients(regression_result: ee.Dictionary) -> dict:
    """
    Extracts coefficients from the regression result and stores them as separate ee.Image objects.

    Args:
        regression_result (ee.Dictionary): The result of the linear regression.

    Returns:
        dict: A dictionary containing the intercept and slope coefficients as ee.Image objects.
    """
    # Extract coefficients array
    coefficients_array = ee.Array(regression_result.get("coefficients"))

    # Convert to list
    coefficients_list = coefficients_array.toList()

    # Extract individual coefficients
    intercept = ee.Image(ee.Number(ee.List(coefficients_list.get(0)).get(0)))
    slope_ndvi = ee.Image(ee.Number(ee.List(coefficients_list.get(1)).get(0)))
    slope_ndbi = ee.Image(ee.Number(ee.List(coefficients_list.get(2)).get(0)))
    slope_ndwi = ee.Image(ee.Number(ee.List(coefficients_list.get(3)).get(0)))

    # Return coefficients as a dictionary
    return {
        "intercept": intercept,
        "slope_ndvi": slope_ndvi,
        "slope_ndbi": slope_ndbi,
        "slope_ndwi": slope_ndwi,
    }


def apply_regression(independent_vars: ee.Image, coefficients: dict) -> ee.Image:
    """
    Applies regression coefficients to an image with independent variables
    to predict the dependent variable.

    Args:
        independent_vars (ee.Image): Image containing bands of independent variables.
        coefficients (dict): Dictionary containing regression coefficients as ee.Image objects.

    Returns:
        ee.Image: Predicted values of the dependent variable.
    """
    # Extract individual coefficients
    intercept = coefficients["intercept"]
    slope_ndvi = coefficients["slope_ndvi"]
    slope_ndbi = coefficients["slope_ndbi"]
    slope_ndwi = coefficients["slope_ndwi"]

    # Apply the regression equation
    predicted = (
        intercept.add(independent_vars.select("NDVI").multiply(slope_ndvi))
        .add(independent_vars.select("NDBI").multiply(slope_ndbi))
        .add(independent_vars.select("NDWI").multiply(slope_ndwi))
    )

    return predicted.rename("predicted_value")


def downscale_lst(
    dependent_vars: ee.Image,
    independent_vars: ee.Image,
    resolution: int,
    s2_indices: ee.Image,
) -> ee.Image:
    """
    Performs LST downscaling using Landsat 8 and Sentinel-2 data.

    Args:
        dependent_vars (ee.Image): Landsat 8 LST image.
        independent_vars (ee.Image): Landsat 8 spectral indices (NDVI, NDBI, NDWI).
        resolution (int): Resolution of the input images in meters.
        s2_indices (ee.Image): Sentinel-2 spectral indices (NDVI, NDBI, NDWI). Must be at 10m resolution.

    Returns:
        ee.Image: Downscaled LST image at Sentinel-2 resolution.
    """
    regression_result = perform_regression(independent_vars, dependent_vars, resolution)

    coefficients = extract_coefficients(regression_result)

    dependent_vars_modeled = apply_regression(dependent_vars, coefficients)

    residuals = compute_residuals(dependent_vars, dependent_vars_modeled)

    smoothed_residuals = apply_gaussian_smoothing(residuals)

    s2_lst_downscaled = apply_regression(s2_indices, coefficients)

    final_downscaled_lst = s2_lst_downscaled.add(smoothed_residuals)

    return final_downscaled_lst.rename("downscaled_LST")
