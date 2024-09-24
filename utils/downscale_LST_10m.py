"""
The logic in this script is described in the paper "Combining Landsat 8 and Sentinel-2 Data in Google Earth Engine to Derive Higher Resolution Land Surface Temperature Maps in Urban Environment" by Katarína Onačillová et al. (2022). DOI: https://doi.org/10.3390/rs14164076

This script was translated to Python by Nicolas Lazaro. 
The original js script written by Silvan Ragettli is at: users/hydrosolutions/CropMapper_India:Scripts4Downscaling/downscale_anything_10m
"""

import ee
from typing import Dict, Any, List, Tuple, Union
from dataclasses import dataclass

# Some options might be redundant. But the naming is so inconsistent that I'm not sure which ones are redundant.
# The options are described here: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
@dataclass
class Sentinel2Options:
    CLOUD_FILTER: int = 60
    CLD_PRB_THRESH: int = 50
    NIR_DRK_THRESH: float = 0.15
    CLD_PRJ_DIST: int = 1
    BUFFER: int = 50
    CC_THRESH: int = 40
    NODATA_THRESH: int = 40
    CC_PIX: int = 50
    MIN_MONTH: int = 1
    MAX_MONTH: int = 12


def cloud_score_sentinel(image: ee.Image) -> ee.Image:
    """Applies cloud scoring to a Sentinel-2 image.

    Args:
        image (ee.Image): Input Sentinel-2 image.

    Returns:
        ee.Image: Cloud-scored image with additional properties.
    """

    def create_mask(image: ee.Image, qa60_condition: ee.ComputedObject) -> ee.Image:
        """Creates a cloud mask based on a QA60 condition.

        Args:
            image (ee.Image): Input image.
            qa60_condition (ee.ComputedObject): Condition applied to the QA60 band.

        Returns:
            ee.Image: The cloud mask image.
        """
        mask = ee.Image(100).where(qa60_condition, 0)
        mask = mask.rename("cloud").updateMask(image.select("B4").gt(0))
        mask = mask.where(image.select("probability").gt(50), 100)
        return mask

    def compute_cloud_pixels(mask: ee.Image, image: ee.Image) -> ee.Number:
        """Computes the mean cloud pixel value over the image footprint.

        Args:
            mask (ee.Image): Cloud mask image.
            image (ee.Image): Input image to get the footprint.

        Returns:
            ee.Number: Mean cloud pixel value.
        """
        return ee.Number(
            mask.select("cloud")
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=image.get("system:footprint"),
                scale=300,
                maxPixels=1e13,
                tileScale=1,
            )
            .get("cloud")
        )

    # Create the first cloud mask
    mask = ee.Image(100)
    condition = image.select("QA60").neq(1024).And(image.select("B4").gt(0))
    mask = mask.where(condition, 0)
    mask = mask.rename("cloud")
    mask = mask.where(image.select("probability").gt(50), 100)

    # Create the second and third cloud masks
    mask2 = create_mask(image, image.select("QA60").neq(1024))
    mask3 = create_mask(image, image.select("QA60").lt(1024))

    # Compute cloud pixels
    cloud_pixels = compute_cloud_pixels(mask, image)
    cloud_pixels2 = compute_cloud_pixels(mask2, image)
    cloud_pixels3 = compute_cloud_pixels(mask3, image)

    # Adjust cloud pixels based on conditions
    cloud_pixels2 = ee.Number(
        ee.Algorithms.If(cloud_pixels.eq(100), 100, cloud_pixels2)
    )
    cloud_pixels3 = ee.Number(
        ee.Algorithms.If(cloud_pixels.eq(100), 100, cloud_pixels3)
    )
    cloud_pixels2 = ee.Number(
        ee.Algorithms.If(cloud_pixels3.gt(80), cloud_pixels3, cloud_pixels2)
    )

    # Return the image with added bands and properties
    return (
        image.updateMask(mask.Not())
        .addBands(mask2)
        .addBands(image.select("probability"), overwrite=True)
        .set(
            {
                "nodata_cover": cloud_pixels,
                "cloud_cover": cloud_pixels2,
                "CLOUD_COVER": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                "SATELLITE": "SENTINEL-2",
            }
        )
    )


def process_image(
    sensing_time: ee.String,
    sentinel2: ee.ImageCollection,
    cloud_prob: ee.ImageCollection,
    aoi: ee.Geometry,
) -> ee.Image:
    """Processes images by combining spectral and cloud probability data.

    Args:
        sensing_time (ee.String): The SENSING_TIME of the image.
        sentinel2 (ee.ImageCollection): Sentinel-2 image collection.
        cloud_prob (ee.ImageCollection): Sentinel-2 cloud probability image collection.
        aoi (ee.Geometry): Area of interest.

    Returns:
        ee.Image: Combined and processed image.
    """
    # Filter by sensing time
    spectral = sentinel2.filter(ee.Filter.eq("SENSING_TIME", sensing_time))
    clouds = cloud_prob.filter(ee.Filter.eq("SENSING_TIME", sensing_time))

    # Mosaic images
    spectral_mosaic = spectral.mosaic()
    cloud_mosaic = clouds.mosaic()

    # Handle cases where cloud probability is missing
    probability_band = ee.Image(
        ee.Algorithms.If(
            clouds.size().gt(0),
            cloud_mosaic.select("probability"),
            ee.Image(0).rename("probability"),
        )
    )

    # Combine spectral and cloud probability images
    combined = spectral_mosaic.addBands(probability_band)

    # Clip to AOI and set properties
    first_image = ee.Image(spectral.first())
    return (
        combined.clip(aoi)
        .copyProperties(first_image)
        .set("system:time_start", first_image.get("system:time_start"))
    )


def get_s2_sr_cld_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date, options: Sentinel2Options
) -> ee.ImageCollection:
    """
    Filter the surface reflectance (SR) and s2cloudless collections according to area of interest and date parameters, then join them on the system:index property. The result is a copy of the SR collection where each image has a new 's2cloudless' property whose value is the corresponding s2cloudless image.

    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (ee.Date): Start date for filtering images.
        end_date (ee.Date): End date for filtering images.

    Returns:
        ee.ImageCollection: Filtered and joined Sentinel-2 SR and s2cloudless image collections.
    """
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", options.CLOUD_FILTER))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )


def add_cloud_bands(image: ee.Image, options: Sentinel2Options) -> ee.Image:
    """Add s2cloudless probability band to the input image.

    Args:
        image (ee.Image): Sentinel-2 image to add cloud bands to.
        options (Sentinel2Options): Sentinel-2 processing options.

    Returns:
        ee.Image: Image with added cloud bands.
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(image.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(options.CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return image.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(image: ee.Image, options: Sentinel2Options) -> ee.Image:
    """Add dark NIR, cloud projection, and identified shadows as image bands.

    Args:
        image (ee.Image): Sentinel-2 image to add shadow bands to.
        options (Sentinel2Options): Sentinel-2 processing options.

    Returns:
        ee.Image: Image with added shadow bands.
    """
    # Identify water pixels from the SCL band.
    not_water = image.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    dark_pixels = (
        image.select("B8")
        .lt(options.NIR_DRK_THRESH)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (
        image.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, options.CLD_PRJ_DIST * 10)
        .reproject(**{"crs": image.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark NIR, cloud projection, and identified shadows to input image.
    return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cloud_shadow_mask(image: ee.Image, options: Sentinel2Options) -> ee.Image:
    """Add cloud and shadow masks to the input image.

    Args:
        image (ee.Image): Sentinel-2 image to add cloud and shadow masks to.
        options (Sentinel2Options): Sentinel-2 processing options.

    Returns:
        ee.Image: Image with added cloud and shadow masks.
    """
    # Add cloud component bands.
    img_cloud = add_cloud_bands(image, options)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud, options)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = (
        img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)
    )

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (
        is_cld_shdw.focalMin(2)
        .focalMax(options.BUFFER * 2 / 20)
        .reproject(**{"crs": image.select([0]).projection(), "scale": 20})
        .rename("cloudmask")
    )

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def add_s2_spectral_indices(image: ee.Image) -> ee.Image:
    """
    Add NDBI, NDWI, and NDVI bands to the image.

    Args:
        image (ee.Image): Sentinel-2 image to add spectral indices to.

    Returns:
        ee.Image: Image with added spectral indices.
    """
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands([ndbi, ndwi, ndvi])


def apply_cloud_shadow_mask(image: ee.Image) -> ee.Image:
    """
    Apply the cloud and shadow mask to the image.

    Args:
        image (ee.Image): Sentinel-2 image to apply the cloud and shadow mask to.

    Returns:
        ee.Image: Image with the cloud and shadow mask applied.
    """
    cloudmask = image.select("cloudmask").Not()
    return image.updateMask(cloudmask)


def get_s2_images(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    options: Dict[str, Any] = None,
) -> ee.ImageCollection:
    """
    Gets Sentinel-2 images for a specific area and time range with custom options.

    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (ee.Date): Start date for filtering images.
        end_date (ee.Date): End date for filtering images.
        options (Dict[str, Any], optional): Dictionary of options to override defaults.

    Returns:
        ee.ImageCollection: Filtered and processed Sentinel-2 image collection.
    """
    # Set options
    s2_options = Sentinel2Options()
    if options:
        for key, value in options.items():
            if hasattr(s2_options, key.upper()):
                setattr(s2_options, key.upper(), value)

    # Get the joined S2 SR and cloud probability collections
    s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date, s2_options)

    def process_image(image: ee.Image) -> ee.Image:
        """Process a single Sentinel-2 image."""
        # Add cloud and shadow bands
        cloud_shadow_image = add_cloud_shadow_mask(image, s2_options)

        # Add spectral indices
        spectral_indices = add_s2_spectral_indices(cloud_shadow_image)

        # Apply cloud and shadow mask
        masked_image = apply_cloud_shadow_mask(spectral_indices)

        return masked_image

    # Process the collection
    processed_collection = s2_sr_cld_col.map(process_image)

    # Filter the processed collection
    final_collection = (
        processed_collection.filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s2_options.CC_PIX)
        )
        .filter(
            ee.Filter.calendarRange(s2_options.MIN_MONTH, s2_options.MAX_MONTH, "month")
        )
        .sort("system:time_start")
    )

    return final_collection


def add_sensing_time(image: ee.Image) -> ee.Image:
    """Adds a 'SENSING_TIME' property to the image in 'YYYY-MM-dd' format.

    Args:
        image (ee.Image): The image to process.

    Returns:
        ee.Image: Image with the 'SENSING_TIME' property added.
    """
    sensing_time = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
    return image.set("SENSING_TIME", sensing_time)


def get_s2_mosaics(
    selected_month: int,
    selected_year: int,
    aoi: ee.Geometry,
    cloud_percentage_threshold: float,
) -> ee.ImageCollection:
    """Creates mosaics of Sentinel-2 images for a specific month and year.

    Args:
        selected_month (int): The month to select images from (1-12).
        selected_year (int): The year to select images from.
        aoi (ee.Geometry): The area of interest.
        cloud_percentage_threshold (float): Maximum allowed CLOUDY_PIXEL_PERCENTAGE.

    Returns:
        ee.ImageCollection: The mosaicked and processed Sentinel-2 image collection.
    """

    def filter_s2_images(collection: ee.ImageCollection) -> ee.ImageCollection:
        """Filters the Sentinel-2 image collection by date, quality, and cloud percentage.

        Args:
            collection (ee.ImageCollection): The Sentinel-2 image collection.

        Returns:
            ee.ImageCollection: Filtered Sentinel-2 image collection.
        """
        return (
            collection.filter(
                ee.Filter.calendarRange(selected_year, selected_year, "year")
            )
            .filter(ee.Filter.calendarRange(selected_month, selected_month, "month"))
            .filter(ee.Filter.eq("GENERAL_QUALITY", "PASSED"))
            .filterBounds(aoi)
            .filterMetadata(
                "CLOUDY_PIXEL_PERCENTAGE", "less_than", cloud_percentage_threshold
            )
            .map(add_sensing_time)
        )

    # Filter Sentinel-2 images and cloud probability images
    sentinel2 = filter_s2_images(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"))
    cloud_prob = filter_s2_images(ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"))

    # Get unique sensing times
    sensing_times = sentinel2.aggregate_array("SENSING_TIME").distinct()

    # Process images and apply cloud scoring
    s2_mosaicked = ee.ImageCollection(
        sensing_times.map(lambda time: process_image(time, sentinel2, cloud_prob, aoi))
    ).map(cloud_score_sentinel)

    return s2_mosaicked


def prepare_image(image: ee.Image, geometry: ee.Geometry) -> ee.Image:
    """Clips the input image to the specified geometry.

    Args:
        image (ee.Image): The image to be clipped.
        geometry (ee.Geometry): The geometry to clip the image to.

    Returns:
        ee.Image: The clipped image.
    """
    return image.clip(geometry)


def calculate_spectral_indices(image: ee.Image) -> ee.Image:
    """Calculates spectral indices (NDVI, NDWI, NDBI) for the input Landsat 8 image.

    Args:
        image (ee.Image): The input Landsat 8 image containing the necessary bands.

    Returns:
        ee.Image: The image with added spectral index bands.
    """
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")

    return image.addBands([ndvi, ndwi, ndbi])


def calculate_lai(image: ee.Image) -> ee.Image:
    """Calculates the Leaf Area Index (LAI) for the input image.

    Args:
        image (ee.Image): The input image with the 'ndvi' band.

    Returns:
        ee.Image: The image with an added 'LAI' band.
    """
    ndvi = image.select("NDVI")

    # Compute LAI. I have no clue what this formula is... lol.
    lai = (
        ndvi.pow(3)
        .multiply(9.519)
        .add(ndvi.pow(2).multiply(0.104))
        .add(ndvi.multiply(1.236))
        .subtract(0.257)
        .rename("LAI")
    )

    return image.addBands(lai)


def prepare_regression_bands(
    l8_image: ee.Image,
    s2_image: ee.Image,
    lst_band: str,
    bands_for_downscaling: List[str],
) -> ee.Image:
    """Prepares the bands required for regression analysis.

    Args:
        l8_image (ee.Image): The Landsat 8 image with spectral indices and LAI.
        s2_image (ee.Image): The Sentinel-2 image with spectral indices and LAI.
        lst_band (str): Name of the LST band in the Landsat 8 image.
        bands_for_downscaling (List[str]): List of band names to use for downscaling.

    Returns:
        ee.Image: An image containing the regression bands.
    """
    l8_lst = l8_image.select(lst_band).rename("L8_LST")
    l8_bands = l8_image.select(bands_for_downscaling)

    # Prepare regression image with a constant band
    regression_image = (
        ee.Image.constant(1).rename("constant").addBands(l8_bands).addBands(l8_lst)
    )

    return regression_image


def perform_linear_regression(
    regression_image: ee.Image, geometry: ee.Geometry, scale: float
) -> Dict[str, ee.Image]:
    """Performs multiple linear regression and extracts coefficients.

    Args:
        regression_image (ee.Image): The image containing regression bands.
        geometry (ee.Geometry): The geometry over which to perform the regression.
        scale (float): The scale (in meters) for the regression.

    Returns:
        Dict[str, ee.Image]: A dictionary mapping band names to their coefficient images.
    """
    # Number of predictor variables (excluding the dependent variable)
    num_bands = regression_image.bandNames().length().subtract(1)

    # Perform the linear regression
    regression = regression_image.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=num_bands, numY=1),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
        tileScale=16,
    )

    # Get the coefficients array
    coefficients = ee.Array(regression.get("coefficients"))

    # Convert coefficients array to image
    coef_image = ee.Image.constant(coefficients.toList().get(0)).rename(
        regression_image.bandNames().slice(0, -1)
    )

    # Create a dictionary of coefficient images
    coef_dict = {
        band: coef_image.select(band)
        for band in regression_image.bandNames().slice(0, -1).getInfo()
    }

    return coef_dict


def calculate_downscaled_lst(
    s2_indices: ee.Image, coefficients: Dict[str, ee.Image]
) -> ee.Image:
    """Calculates the downscaled LST using Sentinel-2 indices and regression coefficients.

    Args:
        s2_indices (ee.Image): Sentinel-2 image with spectral indices.
        coefficients (Dict[str, ee.Image]): Dictionary of regression coefficients.

    Returns:
        ee.Image: The downscaled LST image.
    """
    # Extract intercept and slope coefficients
    intercept = coefficients.get("constant", ee.Image(0))

    # Define the expected band names
    expected_bands = ee.List(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "NDVI", "NDWI"]
    )

    # Rename bands in s2_indices to match the expected names
    renamed_s2_indices = s2_indices.rename(expected_bands)

    # Create a list of images, each representing a term in the regression equation
    regression_terms = expected_bands.map(
        lambda band: renamed_s2_indices.select([band]).multiply(
            coefficients.get(band, ee.Image(0))
        )
    )

    # Sum all terms and add the intercept
    downscaled_lst = ee.ImageCollection(regression_terms).sum().add(intercept)

    return downscaled_lst.rename("downscaled_LST")


def calculate_lst_model_and_residuals(
    l8_lst: ee.Image, l8_indices: ee.Image, coefficients: Dict[str, ee.Image]
) -> Tuple[ee.Image, ee.Image]:
    """Calculates the LST model for Landsat 8 and computes residuals.

    Args:
        l8_lst (ee.Image): Landsat 8 LST image.
        l8_indices (ee.Image): Landsat 8 image with spectral indices.
        coefficients (Dict[str, ee.Image]): Dictionary of regression coefficients.

    Returns:
        Tuple[ee.Image, ee.Image]: A tuple containing the LST model image and residuals image.
    """
    intercept = coefficients["constant"]
    predictors = l8_indices.bandNames()
    slope_images = [coefficients[band] for band in predictors.getInfo()]

    # Compute the LST model
    lst_model = intercept.add(
        ee.ImageCollection(slope_images)
        .toBands()
        .multiply(l8_indices)
        .reduce(ee.Reducer.sum())
    ).rename("LST_model")

    # Compute residuals
    residuals = l8_lst.subtract(lst_model).rename("residuals")

    return lst_model, residuals


def apply_gaussian_convolution(image: ee.Image, radius: float = 1.5) -> ee.Image:
    """Applies Gaussian convolution to the input image.

    Args:
        image (ee.Image): The image to convolve.
        radius (float, optional): Radius of the Gaussian kernel in pixels. Defaults to 1.5.

    Returns:
        ee.Image: The convolved image.
    """
    # Define Gaussian kernel
    gaussian_kernel = ee.Kernel.gaussian(
        radius, sigma=radius, units="pixels", normalize=True
    )

    # Apply convolution
    convolved_image = image.convolve(gaussian_kernel)

    return convolved_image


def generate_final_downscaled_lst(
    downscaled_lst: ee.Image, residuals: ee.Image
) -> ee.Image:
    """Generates the final downscaled LST by adding residuals.

    Args:
        downscaled_lst (ee.Image): The initial downscaled LST image.
        residuals (ee.Image): The residuals image.

    Returns:
        ee.Image: The final downscaled LST image.
    """
    final_lst = downscaled_lst.add(residuals).rename("final_downscaled_LST")
    return final_lst


def generate_downscaled_lst(
    lst_band: str,
    sentinel_bands_for_downscaling: List[str],
    landsat_bands_for_downscaling: List[str],
    initial_scale: float,
    geometry: ee.Geometry,
    l8_image: ee.Image,
    s2_image: ee.Image,
) -> ee.Image:
    """Generates the downscaled Land Surface Temperature (LST) using Landsat 8 and Sentinel-2 data.

    Args:
        lst_band (str): Name of the LST band in the Landsat 8 image.
        sentinel_bands_for_downscaling (List[str]): List of band names to use for downscaling in Sentinel-2.
        landsat_bands_for_downscaling (List[str]): List of band names to use for downscaling in Landsat 8.
        initial_scale (float): Scale (in meters) at which to perform the regression.
        geometry (ee.Geometry): The geometry over which to perform the analysis.
        l8_image (ee.Image): The Landsat 8 image.
        s2_image (ee.Image): The Sentinel-2 image.

    Returns:
        ee.Image: The final downscaled LST image.
    """
    # Prepare images
    l8_prepared = prepare_image(l8_image, geometry)
    s2_prepared = prepare_image(s2_image, geometry)

    # Calculate spectral indices
    l8_indices = calculate_spectral_indices(l8_prepared)

    # Prepare regression bands
    regression_image = prepare_regression_bands(
        l8_indices, s2_prepared, lst_band, landsat_bands_for_downscaling
    )

    # Perform linear regression
    coefficients = perform_linear_regression(regression_image, geometry, initial_scale)

    # Calculate downscaled LST
    s2_bands_for_downscaling = s2_prepared.select(sentinel_bands_for_downscaling)

    downscaled_lst = calculate_downscaled_lst(s2_bands_for_downscaling, coefficients)

    # Calculate LST model and residuals for Landsat 8
    l8_lst = l8_prepared.select(lst_band)
    l8_bands_for_downscaling = l8_indices.select(landsat_bands_for_downscaling)
    lst_model, residuals = calculate_lst_model_and_residuals(
        l8_lst, l8_bands_for_downscaling, coefficients
    )

    # Apply Gaussian convolution to residuals
    smoothed_residuals = apply_gaussian_convolution(residuals)

    # Generate final downscaled LST
    final_downscaled_lst = generate_final_downscaled_lst(
        downscaled_lst, smoothed_residuals
    )

    return final_downscaled_lst
