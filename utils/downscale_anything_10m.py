import ee
from typing import Dict, List, Union, Optional
import logging


class Downscaler:
    """
    A class to perform downscaling of Earth Engine images using regression-based methods.
    """

    def __init__(self, independent_vars: List[str], dependent_var: str):
        """
        Initialize the Downscaler with variable configurations.

        Args:
            independent_vars (List[str]): List of names for independent variables (e.g., ['NDVI', 'NDBI', 'NDWI']).
            dependent_var (str): Name of the dependent variable (e.g., 'ET').
        """
        self.independent_vars = independent_vars
        self.dependent_var = dependent_var
        self.coefficients: Optional[Dict[str, float]] = None
        logging.basicConfig(level=logging.INFO)

    def compute_residuals(
        self, original_image: ee.Image, modeled_image: ee.Image
    ) -> ee.Image:
        """
        Computes the residuals between the original and the modeled image.

        Args:
            original_image (ee.Image): Original image.
            modeled_image (ee.Image): Modeled image based on regression.

        Returns:
            ee.Image: Residuals image.
        """
        return original_image.subtract(modeled_image).rename("residuals")

    def apply_gaussian_smoothing(self, image: ee.Image, radius: float = 1) -> ee.Image:
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
        self,
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
        independent_vars = independent_vars.select(self.independent_vars)
        independent_vars = ee.Image.constant(1).addBands(independent_vars)
        dependent_var = dependent_var.select([self.dependent_var])

        all_vars = independent_vars.addBands(dependent_var)
        numX = ee.List(independent_vars.bandNames()).length()

        try:
            regression = all_vars.reduceRegion(
                reducer=ee.Reducer.linearRegression(numX=numX, numY=1),
                geometry=geometry,
                scale=scale,
                maxPixels=1e13,
                tileScale=16,
            )
            return regression
        except ee.EEException as e:
            logging.error(f"Error in performing regression: {str(e)}")
            raise

    def extract_coefficients(self, regression_result: ee.Dictionary) -> None:
        """
        Extracts coefficients from the regression result and stores them in the class.

        Args:
            regression_result (ee.Dictionary): The result of the linear regression.
        """
        try:
            coefficients = ee.Array(regression_result.get("coefficients")).toList()
            self.coefficients = {
                "intercept": ee.Number(ee.List(coefficients.get(0)).get(0)),
                **{
                    f"slope_{var}": ee.Number(ee.List(coefficients.get(i + 1)).get(0))
                    for i, var in enumerate(self.independent_vars)
                },
            }
        except ee.EEException as e:
            logging.error(f"Error in extracting coefficients: {str(e)}")
            raise

    def apply_regression(self, independent_vars: ee.Image) -> ee.Image:
        """
        Applies the regression coefficients to the independent variables to predict the dependent variable.

        Args:
            independent_vars (ee.Image): Image containing bands of independent variables.

        Returns:
            ee.Image: The predicted dependent variable.
        """
        if not self.coefficients:
            raise ValueError(
                "Coefficients have not been extracted. Run extract_coefficients first."
            )

        try:
            predicted = ee.Image(self.coefficients["intercept"])
            for var in self.independent_vars:
                slope = self.coefficients[f"slope_{var}"]
                predicted = predicted.add(independent_vars.select(var).multiply(slope))

            return predicted.rename("predicted")
        except ee.EEException as e:
            logging.error(f"Error in applying regression: {str(e)}")
            raise

    def downscale(
        self,
        coarse_independent_vars: ee.Image,
        coarse_dependent_var: ee.Image,
        fine_independent_vars: ee.Image,
        geometry: ee.Geometry,
        resolution: int,
    ) -> ee.Image:
        """
        Performs the downscaling process.

        Args:
            coarse_independent_vars (ee.Image): Coarse resolution image with independent variables.
            coarse_dependent_var (ee.Image): Coarse resolution image with dependent variable.
            fine_independent_vars (ee.Image): Fine resolution image with independent variables.
            geometry (ee.Geometry): The geometry over which to perform the downscaling.
            resolution (int): The resolution of the coarse image.

        Returns:
            ee.Image: The downscaled image.
        """
        try:
            fine_projection = fine_independent_vars.projection()
            fine_date = fine_independent_vars.date()
            fine_scale = fine_projection.nominalScale()

            regression_result = self.perform_regression(
                coarse_independent_vars, coarse_dependent_var, geometry, resolution
            )
            self.extract_coefficients(regression_result)

            coarse_modeled = self.apply_regression(coarse_independent_vars)
            residuals = self.compute_residuals(coarse_dependent_var, coarse_modeled)
            smoothed_residuals = self.apply_gaussian_smoothing(residuals)

            fine_downscaled = self.apply_regression(fine_independent_vars)
            final_downscaled = fine_downscaled.add(smoothed_residuals)

            return (
                final_downscaled.rename("downscaled")
                .set("system:time_start", fine_date.millis())
                .setDefaultProjection(fine_projection, None, fine_scale)
            )
        except Exception as e:
            logging.error(f"Error in downscaling process: {str(e)}")
            raise
