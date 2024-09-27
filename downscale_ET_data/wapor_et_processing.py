import ee


def load_wapor_et_data(
    first_year: int, last_year: int, aoi: ee.Geometry
) -> ee.ImageCollection:
    """
    Load and process WAPOR ET dekadal data for a range of years.

    Args:
        first_year (int): The first year to process.
        last_year (int): The last year to process.
        aoi (ee.Geometry): The buffered area of interest.

    Returns:
        ee.ImageCollection: Processed WAPOR ET dekadal data.
    """

    # Define the target projection (EPSG:4326)
    target_projection = ee.Projection("EPSG:4326")

    def process_dekad(dekad, yr):
        # Calculate month and dekad within month
        month = ee.Number(dekad).add(2).divide(3).ceil().clamp(1, 12).int()
        dekad_in_month = ee.Number(dekad).mod(3).add(1).int()

        # Construct the URL for the GeoTIFF file
        url = ee.String(
            "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-D/WAPOR-3.L1-AETI-D."
        )
        url = (
            url.cat(ee.Number(yr).format("%04d"))
            .cat("-")
            .cat(ee.Number(month).format("%02d"))
            .cat("-D")
            .cat(ee.Number(dekad_in_month).format())
            .cat(".tif")
        )

        # Load and process the image
        return (
            ee.Image.loadGeoTIFF(url)
            .multiply(0.1)
            .int()
            .set("Dekad", dekad)
            .set("Year", yr)
            .set(
                "system:time_start",
                ee.Date.fromYMD(yr, 1, 1)
                .advance(ee.Number(dekad).subtract(1).multiply(10), "day")
                .millis(),
            )
            .clip(aoi)
            .rename("ET")
        )

    def process_year(yr):
        return ee.List.sequence(1, 36).map(lambda dekad: process_dekad(dekad, yr))

    # Process main years
    wapor_eta_dekadal = ee.ImageCollection(
        ee.List.sequence(first_year, last_year).map(process_year).flatten()
    )

    return wapor_eta_dekadal
