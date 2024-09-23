import ee
from typing import Dict, Any, List, Tuple, Union


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
        .addBands({"srcImg": image.select("probability"), "overwrite": True})
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


def get_s2_images(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date, options: Dict[str, Any]
) -> ee.ImageCollection:
    """Gets Sentinel-2 images for a specific area and time range with custom options.

    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (ee.Date): Start date for filtering images.
        end_date (ee.Date): End date for filtering images.
        options (Dict[str, Any]): Dictionary of options including:
            - cc_thresh (int): Cloud cover threshold.
            - nodata_thresh (int): No-data cover threshold.
            - cc_pix (int): Cloudy pixel percentage threshold.
            - min_m (int): Minimum month for filtering.
            - max_m (int): Maximum month for filtering.

    Returns:
        ee.ImageCollection: Filtered and processed Sentinel-2 image collection.
    """
    # Extract options with default values
    cc_thresh = options.get("cc_thresh", 40)
    nodata_thresh = options.get("nodata_thresh", 40)
    cc_pix = options.get("cc_pix", 50)
    min_month = options.get("min_m", 1)
    max_month = options.get("max_m", 12)

    def add_bands(image: ee.Image) -> ee.Image:
        """Adds MNDWI, NDWI, and NDVI bands to the image.

        Args:
            image (ee.Image): The Sentinel-2 image.

        Returns:
            ee.Image: Image with additional bands and SENSING_TIME property.
        """
        mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        ndwi = image.normalizedDifference(["B4", "B11"]).rename("NDWI")
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        bands = image.select(["B2", "B3", "B4", "B8", "B11", "B12", "QA60"])
        sensing_time = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return bands.addBands([mndwi, ndvi, ndwi]).set("SENSING_TIME", sensing_time)

    # Load and process Sentinel-2 imagery
    sentinel2 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(min_month, max_month, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cc_pix))
        .map(add_bands)
    )

    def set_sensing_time(image: ee.Image) -> ee.Image:
        """Sets the SENSING_TIME property based on system:time_start.

        Args:
            image (ee.Image): The image to process.

        Returns:
            ee.Image: Image with SENSING_TIME property set.
        """
        sensing_time = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return image.set("SENSING_TIME", sensing_time)

    # Load and process Sentinel-2 cloud probability data
    cloud_prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .map(set_sensing_time)
    )

    # Get unique sensing times
    sensing_times = sentinel2.aggregate_array("SENSING_TIME").distinct()

    # Process each image by sensing time
    processed_images = ee.ImageCollection(
        sensing_times.map(lambda time: process_image(time, sentinel2, cloud_prob, aoi))
    )

    # TODO: reflect if clipping is necessary
    def apply_cloud_scoring(image: ee.Image) -> ee.Image:
        """Applies cloud scoring and clips the image to the AOI.

        Args:
            image (ee.Image): The image to process.

        Returns:
            ee.Image: The cloud-scored image.
        """
        return cloud_score_sentinel(image.clip(aoi))

    # Apply cloud scoring and filter images
    final_collection = (
        processed_images.map(apply_cloud_scoring)
        .filter(ee.Filter.lt("cloud_cover", cc_thresh))
        .filter(ee.Filter.lt("nodata_cover", nodata_thresh))
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
    """Calculates spectral indices (NDVI, NDWI, NDBI) for the input image.

    Args:
        image (ee.Image): The input image containing the necessary bands.

    Returns:
        ee.Image: The image with added spectral index bands.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndwi = image.normalizedDifference(["B3", "B11"]).rename("ndwi")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("ndbi")

    return image.addBands([ndvi, ndwi, ndbi])


def calculate_lai(image: ee.Image) -> ee.Image:
    """Calculates the Leaf Area Index (LAI) for the input image.

    Args:
        image (ee.Image): The input image with the 'ndvi' band.

    Returns:
        ee.Image: The image with an added 'LAI' band.
    """
    ndvi = image.select("ndvi")

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
    intercept = coefficients["constant"]
    predictors = s2_indices.bandNames()
    slope_images = [coefficients[band] for band in predictors.getInfo()]

    # Compute the downscaled LST
    downscaled_lst = intercept.add(
        ee.ImageCollection(slope_images)
        .toBands()
        .multiply(s2_indices)
        .reduce(ee.Reducer.sum())
    )

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
    bands_for_downscaling: List[str],
    initial_scale: float,
    geometry: ee.Geometry,
    l8_image: ee.Image,
    s2_image: ee.Image,
) -> ee.Image:
    """Generates the downscaled Land Surface Temperature (LST) using Landsat 8 and Sentinel-2 data.

    Args:
        lst_band (str): Name of the LST band in the Landsat 8 image.
        bands_for_downscaling (List[str]): List of band names to use for downscaling.
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
    s2_indices = calculate_spectral_indices(s2_prepared)

    # Calculate LAI
    l8_with_lai = calculate_lai(l8_indices)
    s2_with_lai = calculate_lai(s2_indices)

    # Prepare regression bands
    regression_image = prepare_regression_bands(
        l8_with_lai, s2_with_lai, lst_band, bands_for_downscaling
    )

    # Perform linear regression
    coefficients = perform_linear_regression(regression_image, geometry, initial_scale)

    # Calculate downscaled LST
    s2_bands_for_downscaling = s2_with_lai.select(bands_for_downscaling)
    downscaled_lst = calculate_downscaled_lst(s2_bands_for_downscaling, coefficients)

    # Calculate LST model and residuals for Landsat 8
    l8_lst = l8_prepared.select(lst_band)
    l8_bands_for_downscaling = l8_with_lai.select(bands_for_downscaling)
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
