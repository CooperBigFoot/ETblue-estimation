import ee


def load_wapor_et_data(
    first_year: int, last_year: int, frequency: str = "dekadal"
) -> ee.ImageCollection:
    """
    Load and process WAPOR ET data for a range of years with specified frequency.

    Args:
        first_year (int): The first year to process.
        last_year (int): The last year to process.
        frequency (str): The frequency of data to load. Either "dekadal" or "monthly".

    Returns:
        ee.ImageCollection: Processed WAPOR ET data.
    """

    def process_dekadal(dekad, yr):
        month = ee.Number(dekad).add(2).divide(3).ceil().clamp(1, 12).int()

        dekad_in_month = ee.Number(dekad).mod(3).add(1).int()

        url = (
            ee.String(
                "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-D/WAPOR-3.L1-AETI-D."
            )
            .cat(ee.Number(yr).format("%04d"))
            .cat("-")
            .cat(ee.Number(month).format("%02d"))
            .cat("-D")
            .cat(ee.Number(dekad_in_month).format("%d"))
            .cat(".tif")
        )

        return (
            ee.Image.loadGeoTIFF(url)
            .multiply(0.1)
            .int()
            .set(
                "system:time_start",
                ee.Date.fromYMD(yr, month, 1)
                .advance(ee.Number(dekad).subtract(1).multiply(10), "day")
                .millis(),
            )
            .set("Month", month)
            .set("Year", yr)
            .rename("ET")
        )

    def process_monthly(month, yr):
        url = (
            ee.String(
                "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-M/WAPOR-3.L1-AETI-M."
            )
            .cat(ee.Number(yr).format("%04d"))
            .cat("-")
            .cat(ee.Number(month).format("%02d"))
            .cat(".tif")
        )

        return (
            ee.Image.loadGeoTIFF(url)
            .multiply(0.1)
            .int()
            .set(
                "system:time_start",
                ee.Date.fromYMD(yr, month, 1).millis(),
            )
            .set("Month", month)
            .set("Year", yr)
            .rename("ET")
        )

    if frequency == "dekadal":
        collection = ee.ImageCollection(
            ee.List.sequence(first_year, last_year)
            .map(
                lambda yr: ee.List.sequence(0, 35).map(
                    lambda dekad: process_dekadal(dekad, yr)
                )
            )
            .flatten()
        )
    elif frequency == "monthly":
        collection = ee.ImageCollection(
            ee.List.sequence(first_year, last_year)
            .map(
                lambda yr: ee.List.sequence(1, 12).map(
                    lambda month: process_monthly(month, yr)
                )
            )
            .flatten()
        )
    else:
        raise ValueError("Frequency must be either 'dekadal' or 'monthly'")

    scale = collection.first().projection().nominalScale()

    return collection.sort("system:time_start")
