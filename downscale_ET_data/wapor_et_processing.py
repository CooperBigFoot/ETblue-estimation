import ee



# TODO: aoi should be ee.Geometry and not ee.FeatureCollection
def load_wapor_et_data(
    first_year: int, last_year: int, aoi: ee.Geometry
) -> ee.ImageCollection:
    """
    Load and process WAPOR ET data for a range of years.

    Args:
        first_year (int): The first year to process.
        last_year (int): The last year to process.
        aoi (ee.FeatureCollection): The buffered area of interest.

    Returns:
        ee.ImageCollection: Processed WAPOR ET data.
    """

    def process_month(mm, yr):
        # Construct the URL for the GeoTIFF file
        url = ee.String(
            "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-M/WAPOR-3.L1-AETI-M."
        )
        url = (
            url.cat(ee.Number(yr).format("%04d"))
            .cat("-")
            .cat(ee.Number(mm).format("%02d"))
            .cat(".tif")
        )

        # Load and process the image
        return (
            ee.Image.loadGeoTIFF(url)
            .multiply(0.1)
            .int()
            .set("Month", mm)
            .set("Year", yr)
            .set("system:time_start", ee.Date.fromYMD(yr, mm, 1).millis())
            .rename("ET")
            .clip(aoi)
        )

    def process_year(yr):
        return ee.List.sequence(1, 12).map(lambda mm: process_month(mm, yr))

    # Process main years
    wapor_eta_monthly = ee.ImageCollection(
        ee.List.sequence(first_year, last_year).map(process_year).flatten()
    )

    # Process first 6 months of the following year
    next_year = last_year + 1
    additional_months = ee.List.sequence(1, 6).map(
        lambda mm: process_month(mm, next_year)
    )
    wapor_eta_monthly = wapor_eta_monthly.merge(ee.ImageCollection(additional_months))

    return wapor_eta_monthly
