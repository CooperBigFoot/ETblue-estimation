import ee
from typing import Union
from filter_nutzungsflaechen import get_crops_to_exclude, get_rainfed_reference_crops


def compute_et_blue_value(
    feature: ee.Feature,
    et_blue_image: ee.Image,
    output_property: str = "ET_blue_m3_ha",
    scale: int = 10,
    threshold: Union[int, float] = 0,  # New parameter
) -> ee.Feature:
    """
    Compute ET blue value per hectare for a single feature.
    Uses median check to ensure consistent irrigation signal across the field.
    Values below threshold are set to 0.

    Args:
        feature: Input field feature
        et_blue_image: Image with ET blue values in m3
        output_property: Name of the property to store the computed ET blue value
        scale: Scale in meters for computation
        threshold: Minimum value in m³/ha, below which a field is considered non-irrigated

    Returns:
        Feature with added ET blue property in m³/ha
    """
    geometry = feature.geometry()
    area_ha = ee.Number(geometry.area()).divide(10000)

    # Check if feature's NUTZUNG is in excluded list
    excluded_crops = get_crops_to_exclude()
    rainfed_crops = get_rainfed_reference_crops()
    is_excluded = ee.List([*excluded_crops, *rainfed_crops]).contains(
        feature.get("nutzung")
    )

    def calculate_et_blue():
        # First check if median ET blue is positive (consistent irrigation signal)
        median_et_blue = et_blue_image.reduceRegion(
            reducer=ee.Reducer.median(), geometry=geometry, scale=scale, maxPixels=1e9
        ).get("ET_blue_m3")

        # Only calculate total if median indicates consistent irrigation
        def compute_total():
            total = et_blue_image.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geometry, scale=scale, maxPixels=1e9
            ).get("ET_blue_m3")

            value = ee.Number(
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(total, None),
                    0,
                    ee.Number(total).divide(area_ha).round(),
                )
            )

            # Apply threshold
            return ee.Number(ee.Algorithms.If(value.lt(threshold), 0, value))

        return ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(median_et_blue, None),
                0,
                ee.Algorithms.If(ee.Number(median_et_blue).gt(0), compute_total(), 0),
            )
        )

    et_blue_value = ee.Number(ee.Algorithms.If(is_excluded, 0, calculate_et_blue()))

    return feature.set(output_property, et_blue_value)


def calculate_et_blue_per_field(
    et_blue_image: ee.Image,
    crop_fields: ee.FeatureCollection,
    output_property: str = "ET_blue_m3_ha",
    scale: Union[int, float] = 10,
    threshold: Union[int, float] = 0,
) -> ee.FeatureCollection:
    """
    Calculate ET_blue in m3/ha for each crop field.

    Args:
        et_blue_image: Image containing ET_blue estimates in m3
        crop_fields: Collection of crop field features
        output_property: Name of the property to store the computed ET blue value
        scale: Scale in meters for computations
        threshold: Minimum value in m³/ha, below which a field is considered non-irrigated

    Returns:
        FeatureCollection with ET blue property added to all features
    """
    return crop_fields.map(
        lambda f: compute_et_blue_value(
            f, et_blue_image, output_property, scale, threshold
        )
    )


def process_image_collection(
    image_collection: ee.ImageCollection,
    feature_collection: ee.FeatureCollection,
    output_prefix: str,
    asset_path: str,
) -> None:
    """
    Process each image in a collection against a feature collection.

    Args:
        image_collection: Collection of images to process
        feature_collection: Collection of features to process against each image
        output_prefix: Prefix for the output task names and asset IDs
        asset_path: Base path for the output assets
    """
    # Convert image collection to list for processing
    image_list = image_collection.toList(image_collection.size())
    collection_size = image_list.size().getInfo()

    for i in range(collection_size):
        image = ee.Image(image_list.get(i))

        # Process the image
        fields_with_et = calculate_et_blue_per_field(image, feature_collection)

        # Get image date for naming
        image_id = image.id().getInfo()

        # Create export task
        task_name = f"{output_prefix}_{image_id}"
        asset_id = f"{asset_path}/{task_name}"

        export_feature_collection(fields_with_et, task_name, asset_id)
        print(f"Exporting {task_name} to {asset_id}")


def export_feature_collection(
    collection: ee.FeatureCollection, task_name: str, asset_id: str
):
    """
    Export the feature collection to an Earth Engine asset.

    Args:
        collection: The feature collection to export
        task_name: The name of the export task
        asset_id: The asset ID to export to
    """
    task = ee.batch.Export.table.toAsset(
        collection=collection,
        description=task_name,
        assetId=asset_id,
    )
    task.start()
