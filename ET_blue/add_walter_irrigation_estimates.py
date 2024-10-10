import ee


def filter_and_estimate_irrigation(
    feature_collection: ee.FeatureCollection,
) -> ee.FeatureCollection:
    """
    Filter crops mentioned by Walter Koch and add estimated irrigation values.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection.

    Returns:
        ee.FeatureCollection: Filtered feature collection with estimated irrigation values.
    """
    # Define the crops mentioned by the farmer and their irrigation ranges
    crop_irrigation = ee.Dictionary(
        {
            "Einjährige Freilandgemüse, ohne Konservengemüse": [200, 1000],
            "Kartoffeln": [200, 1000],
            "Freiland-Konservengemüse": [200, 600],
        }
    )

    # Convert the dictionary keys to an ee.List
    crop_list = crop_irrigation.keys()

    def add_irrigation_estimate(feature):
        nutzung = feature.get("nutzung")

        def set_irrigation_range(crop_type):
            range_values = ee.List(crop_irrigation.get(crop_type))
            min_val = ee.Number(range_values.get(0))
            max_val = ee.Number(range_values.get(1))
            return feature.set(
                "estimated_irrigation_m3_ha_year",
                ee.String(min_val.format()).cat("-").cat(ee.String(max_val.format())),
            )

        return ee.Algorithms.If(
            crop_list.contains(nutzung), set_irrigation_range(nutzung), feature
        )

    # Filter the feature collection to include only the specified crops
    # and add the estimated irrigation field
    filtered_collection = feature_collection.filter(
        ee.Filter.inList("nutzung", crop_list)
    ).map(add_irrigation_estimate)

    return filtered_collection
