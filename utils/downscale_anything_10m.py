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
    image_resolution: int,
    geometry: ee.Geometry,
) -> ee.Dictionary:
    """
    Performs linear regression using independent variables to predict the dependent variable.

    Args:
        independent_vars (ee.Image): Image containing bands of independent variables. It should have three bands: fitted_NDVI, fitted_NDBI, and fitted_NDWI.
        dependent_var (ee.Image): Single-band image of the dependent variable.
        image_resolution (int): Resolution of the input images in meters.
        geometry (ee.Geometry): The geometry over which to perform the regression.

    Returns:
        ee.Dictionary: The result of the linear regression.
    """
    # Ensure the independent_vars image has the correct bands
    independent_vars = independent_vars.select(
        ["fitted_NDVI", "fitted_NDBI", "fitted_NDWI"]
    )

    # Ensure the dependent_var image has the correct band
    dependent_var = dependent_var.select(["ET"])

    all_vars = independent_vars.addBands(dependent_var)

    independent_names = independent_vars.bandNames()
    full_regression = all_vars.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=independent_names.length(), numY=1),
        geometry=geometry,
        scale=image_resolution,
        maxPixels=1e13,
    )

    return full_regression


def extract_coefficients(regression_result: ee.Dictionary) -> Dict:
    """
    Extracts coefficients from the regression result and stores them as separate ee.Image objects.

    Args:
        regression_result (ee.Dictionary): The result of the linear regression.

    Returns:
        dict: A dictionary containing the intercept and slope coefficients as ee.Image objects.
    """
    coefficients = regression_result.get("coefficients")

    coefficients_array = ee.Array(coefficients)

    coefficients_list = coefficients_array.toList()

    # Extract individual coefficients
    intercept = ee.Image(ee.Number(ee.List(coefficients_list.get(0)).get(0)))
    slope_fitted_ndvi = ee.Image(ee.Number(ee.List(coefficients_list.get(1)).get(0)))
    slope_fitted_ndbi = ee.Image(ee.Number(ee.List(coefficients_list.get(2)).get(0)))
    slope_fitted_ndwi = ee.Image(ee.Number(ee.List(coefficients_list.get(3)).get(0)))

    # Return coefficients as a dictionary
    return {
        "intercept": intercept,
        "slope_fitted_ndvi": slope_fitted_ndvi,
        "slope_fitted_ndbi": slope_fitted_ndbi,
        "slope_fitted_ndwi": slope_fitted_ndwi,
    }


def apply_regression(independent_vars: ee.Image, coefficients: Dict) -> ee.Image:
    """
    Applies regression coefficients to an image with independent variables
    to predict the dependent variable.

    Args:
        independent_vars (ee.Image): Image containing bands of independent variables.
        coefficients (dict): Dictionary containing regression coefficients as ee.Image objects.

    Returns:
        ee.Image: Predicted values of the dependent variable.
    """
    # Ensure the independent_vars image has the correct bands
    independent_vars = independent_vars.select(
        ["fitted_NDVI", "fitted_NDBI", "fitted_NDWI"]
    )

    # Extract individual coefficients
    intercept = coefficients["intercept"]
    slope_fitted_ndvi = coefficients["slope_fitted_ndvi"]
    slope_fitted_ndbi = coefficients["slope_fitted_ndbi"]
    slope_fitted_ndwi = coefficients["slope_fitted_ndwi"]

    # Apply the regression equation
    predicted = (
        intercept.add(
            independent_vars.select("fitted_NDVI").multiply(slope_fitted_ndvi)
        )
        .add(independent_vars.select("fitted_NDBI").multiply(slope_fitted_ndbi))
        .add(independent_vars.select("fitted_NDWI").multiply(slope_fitted_ndwi))
    )

    return predicted.rename("predicted_value")


def downscale(
    dependent_vars: ee.Image,
    independent_vars: ee.Image,
    resolution: int,
    s2_indices: ee.Image,
    geometry: ee.Geometry,
) -> ee.Image:
    """
    Performs downscaling.

    Args:
        dependent_vars (ee.Image): Landsat 8 image.
        independent_vars (ee.Image): Landsat 8 spectral indices (fitted_NDVI, fitted_NDBI, fitted_NDWI).
        resolution (int): Resolution of the input images in meters.
        s2_indices (ee.Image): Sentinel-2 spectral indices (fitted_NDVI, fitted_NDBI, fitted_NDWI). Must be at 10m resolution.
        geometry (ee.Geometry): The geometry over which to perform the downscaling.

    Returns:
        ee.Image: Downscaled image at Sentinel-2 resolution.
    """
    regression_result = perform_regression(
        independent_vars, dependent_vars, resolution, geometry
    )

    coefficients = extract_coefficients(regression_result)

    dependent_vars_modeled = apply_regression(independent_vars, coefficients)

    residuals = compute_residuals(dependent_vars, dependent_vars_modeled)

    smoothed_residuals = apply_gaussian_smoothing(residuals)

    s2_downscaled = apply_regression(s2_indices, coefficients)

    final_downscaled = s2_downscaled.add(smoothed_residuals)

    return final_downscaled.rename("downscaled")
