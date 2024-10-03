import ee
import time
from typing import List, Dict, Any


def export_image_to_asset(
    image: ee.Image,
    region: ee.Geometry,
    project_name: str,
    asset_folder: str,
    scale: int = 30,
    max_pixels: int = 1e13,
) -> ee.batch.Task:
    """
    Create an export task for a single image.

    Args:
        image (ee.Image): Image to export.
        region (ee.Geometry): Region to export.
        project_name (str): Name of your GEE project.
        asset_folder (str): Name of the asset folder to export to.
        scale (int, optional): Scale in meters. Defaults to 30.
        max_pixels (int, optional): Maximum number of pixels to export. Defaults to 1e13.

    Returns:
        ee.batch.Task: Export task.
    """
    asset_id = f"projects/{project_name}/assets/{asset_folder}/{image.id().getInfo()}"

    return ee.batch.Export.image.toAsset(
        image=image,
        description=f"Export_{image.id().getInfo()}",
        assetId=asset_id,
        scale=scale,
        region=region,
        maxPixels=max_pixels,
    )


def export_collection_to_assets(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    project_name: str,
    asset_folder: str,
    scale: int = 30,
    max_concurrent: int = 3,
    delay: int = 60,
) -> None:
    """
    Export an entire image collection to assets.

    Args:
        collection (ee.ImageCollection): Image collection to export.
        region (ee.Geometry): Region to export.
        project_name (str): Name of your GEE project.
        asset_folder (str): Name of the asset folder to export to.
        max_concurrent (int, optional): Maximum number of concurrent export tasks. Defaults to 3.
        delay (int, optional): Delay in seconds between starting tasks. Defaults to 60.
    """
    image_list = collection.toList(collection.size())
    size = image_list.size().getInfo()

    active_tasks: List[ee.batch.Task] = []

    for i in range(size):
        image = ee.Image(image_list.get(i))
        task = export_image_to_asset(image, region, project_name, asset_folder, scale=scale)

        while len(active_tasks) >= max_concurrent:
            active_tasks = [task for task in active_tasks if task.active()]
            time.sleep(delay)

        task.start()
        active_tasks.append(task)
        print(f"Started export task for image {i+1} of {size}")
        time.sleep(delay)

    print("All export tasks have been started.")
