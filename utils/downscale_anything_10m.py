"""
The logic in this script is inspired from the paper "Combining Landsat 8 and Sentinel-2 Data in Google Earth Engine to Derive Higher Resolution Land Surface Temperature Maps in Urban Environment" by Katarína Onačillová et al. (2022). DOI: https://doi.org/10.3390/rs14164076

This script was written by Nicolas Lazaro. 
The original js script written by Silvan Ragettli is at: users/hydrosolutions/CropMapper_India:Scripts4Downscaling/downscale_anything_10m
"""

import ee
from typing import Dict


def compute_residuals(original_image: ee.Image, modeled_image: ee.Image) -> ee.Image:
    """
    Computes the residuals between the original and the modeled.

    Args:
        original_image (ee.Image): Original Landsat 8 image.
        modeled_image (ee.Image): Modeled based on regression.

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
    independent_vars: ee.Image,
    dependent_var: ee.Image,
    geometry: ee.Geometry,
    scale: float,
) -> ee.Dictionary:
    """
    Performs linear regression using independent variables to predict the dependent variable.

    Args:
        independent_vars (ee.Image): Image containing bands of independent variables.
        dependent_var (ee.Image): Single-band image of the dependent variable.
        geometry (ee.Geometry): The geometry over which to perform the regression.
        scale (float): The scale at which to perform the regression.

    Returns:
        ee.Dictionary: The result of the linear regression.
    """
    # Ensure the independent_vars image has the correct bands
    independent_vars = independent_vars.select(
        ["fitted_NDVI", "fitted_NDBI", "fitted_NDWI"]
    )

    independent_vars = ee.Image.constant(1).addBands(independent_vars)

    # Ensure the dependent_var image has the correct band
    dependent_var = dependent_var.select(["ET"])

    # Combine independent and dependent variables
    all_vars = independent_vars.addBands(dependent_var)

    # Perform the regression
    regression = all_vars.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=4, numY=1),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
        tileScale=16,
    )

    return regression


def extract_coefficients(regression_result: ee.Dictionary) -> Dict[str, ee.Image]:
    """
    Extracts coefficients from the regression result and stores them as separate ee.Image objects.

    Args:
        regression_result (ee.Dictionary): The result of the linear regression.

    Returns:
        Dict[str, ee.Image]: A dictionary containing the intercept and slope coefficients as ee.Image objects.
    """

    regression_result = ee.Dictionary(regression_result)
    coefficients = ee.Array(regression_result.get("coefficients"))
    return {
        "intercept": ee.Image(coefficients.get([0, 0])),
        "slope_fitted_ndvi": ee.Image(coefficients.get([1, 0])),
        "slope_fitted_ndbi": ee.Image(coefficients.get([2, 0])),
        "slope_fitted_ndwi": ee.Image(coefficients.get([3, 0])),
    }


def apply_regression(
    independent_vars: ee.Image, coefficients: ee.Dictionary
) -> ee.Image:
    independent_vars = independent_vars.select(
        ["fitted_NDVI", "fitted_NDBI", "fitted_NDWI"]
    )
    # Multiply each independent variable by its corresponding coefficient
    predicted = ee.Image(coefficients.get("intercept")).add(
        independent_vars.multiply(
            ee.Image.cat(
                [
                    ee.Image(coefficients.get("slope_fitted_ndvi")),
                    ee.Image(coefficients.get("slope_fitted_ndbi")),
                    ee.Image(coefficients.get("slope_fitted_ndwi")),
                ]
            )
        ).reduce("sum")
    )
    return predicted.rename("predicted_value")


def downscale(
    dependent_vars: ee.Image,
    independent_vars: ee.Image,
    resolution: int,
    s2_indices: ee.Image,
    geometry: ee.Geometry,
) -> ee.Image:
    regression_result = perform_regression(
        independent_vars, dependent_vars, geometry, resolution
    )

    # Use ee.Algorithms.If to handle potential null results
    coefficients = extract_coefficients(regression_result)

    dependent_vars_modeled = apply_regression(
        independent_vars, ee.Dictionary(coefficients)
    )
    residuals = compute_residuals(dependent_vars, dependent_vars_modeled)
    smoothed_residuals = apply_gaussian_smoothing(residuals)

    # Ensure consistent resolution and projection
    s2_projection = s2_indices.projection()
    s2_downscaled = apply_regression(s2_indices, ee.Dictionary(coefficients)).reproject(
        s2_projection
    )
    smoothed_residuals = smoothed_residuals.reproject(s2_projection)

    final_downscaled = s2_downscaled.add(smoothed_residuals)
    s2_date = s2_indices.date()
    scale = s2_projection.nominalScale()

    final_downscaled_with_metadata = (
        final_downscaled.rename("downscaled")
        .set("system:time_start", s2_date.millis())
        .setDefaultProjection(s2_projection, None, scale)
    )
    return final_downscaled_with_metadata
