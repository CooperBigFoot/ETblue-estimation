# File: /src/gee_processing/export_utils.py

import ee
import time
from typing import List, Dict, Any, Optional
from datetime import datetime


def create_asset_id(
    image: ee.Image, project_name: str, asset_folder: str, prefix: Optional[str] = None
) -> str:
    """
    Create a properly formatted asset ID with optional prefix.

    Args:
        image (ee.Image): Image to export
        project_name (str): GEE project name
        asset_folder (str): Asset folder name
        prefix (Optional[str]): Optional prefix for the asset name

    Returns:
        str: Properly formatted asset ID
    """
    try:
        # Get image ID or system timestamp if no ID exists
        img_id = image.get("system:id").getInfo()
        if img_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_id = f"image_{timestamp}"

        # Clean asset folder path
        asset_folder = asset_folder.strip("/")

        # Add prefix if provided
        if prefix:
            img_id = f"{prefix}_{img_id}"

        return f"projects/{project_name}/assets/{asset_folder}/{img_id}"
    except ee.EEException as e:
        raise ValueError(f"Error creating asset ID: {str(e)}")


def export_image_to_asset(
    image: ee.Image,
    region: ee.Geometry,
    project_name: str,
    asset_folder: str,
    description: Optional[str] = None,
    scale: int = 30,
    max_pixels: int = 1e13,
    prefix: Optional[str] = None,
) -> ee.batch.Task:
    """
    Create an export task for a single image with improved error handling and naming.

    Args:
        image (ee.Image): Image to export
        region (ee.Geometry): Region to export
        project_name (str): Name of your GEE project
        asset_folder (str): Name of the asset folder to export to
        description (Optional[str]): Custom description for the task
        scale (int): Scale in meters. Defaults to 30
        max_pixels (int): Maximum number of pixels to export. Defaults to 1e13
        prefix (Optional[str]): Optional prefix for the asset name

    Returns:
        ee.batch.Task: Export task

    Raises:
        ValueError: If required parameters are invalid
        ee.EEException: If GEE operations fail
    """
    # Input validation
    if not isinstance(image, ee.Image):
        raise ValueError("Input must be an ee.Image object")
    if scale <= 0:
        raise ValueError("Scale must be positive")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    try:
        asset_id = create_asset_id(image, project_name, asset_folder, prefix)

        # Create default description if none provided
        if description is None:
            description = f"Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ee.batch.Export.image.toAsset(
            image=image,
            description=description,
            assetId=asset_id,
            scale=scale,
            region=region,
            maxPixels=max_pixels,
        )
    except ee.EEException as e:
        raise ee.EEException(f"Export task creation failed: {str(e)}")


def monitor_tasks(
    tasks: List[ee.batch.Task],
    polling_interval: int = 30,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Monitor the status of export tasks with timeout and detailed reporting.

    Args:
        tasks (List[ee.batch.Task]): List of export tasks to monitor
        polling_interval (int): Seconds between status checks. Defaults to 30
        timeout (Optional[int]): Maximum seconds to wait. None for no timeout

    Returns:
        Dict[str, Any]: Status report for all tasks
    """
    start_time = time.time()
    active_tasks = tasks.copy()
    results = {"completed": [], "failed": [], "timed_out": [], "duration": 0}

    while active_tasks:
        if timeout and (time.time() - start_time) > timeout:
            results["timed_out"] = [
                task.status()["description"] for task in active_tasks
            ]
            break

        for task in active_tasks[:]:  # Create copy for safe iteration
            status = task.status()
            state = status["state"]

            if state == "COMPLETED":
                results["completed"].append(status["description"])
                active_tasks.remove(task)
            elif state == "FAILED":
                results["failed"].append(
                    {
                        "description": status["description"],
                        "error_message": status.get("error_message", "Unknown error"),
                    }
                )
                active_tasks.remove(task)

        if active_tasks:
            time.sleep(polling_interval)

    results["duration"] = time.time() - start_time
    return results


def export_collection_to_assets(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    project_name: str,
    asset_folder: str,
    scale: int = 30,
    max_concurrent: int = 3,
    prefix: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Export an image collection to assets with improved task management and monitoring.

    Args:
        collection (ee.ImageCollection): Image collection to export
        region (ee.Geometry): Region to export
        project_name (str): Name of your GEE project
        asset_folder (str): Name of the asset folder to export to
        scale (int): Scale in meters. Defaults to 30
        max_concurrent (int): Maximum number of concurrent export tasks. Defaults to 3
        prefix (Optional[str]): Optional prefix for asset names
        timeout (Optional[int]): Maximum seconds to wait for all exports. None for no timeout

    Returns:
        Dict[str, Any]: Export results summary

    Raises:
        ValueError: If input parameters are invalid
        ee.EEException: If GEE operations fail
    """
    if not isinstance(collection, ee.ImageCollection):
        raise ValueError("Input must be an ee.ImageCollection object")

    try:
        image_list = collection.toList(collection.size())
        size = image_list.size().getInfo()

        if size == 0:
            return {"warning": "Collection is empty", "tasks_started": 0}

        active_tasks: List[ee.batch.Task] = []
        all_tasks: List[ee.batch.Task] = []

        for i in range(size):
            image = ee.Image(image_list.get(i))

            # Wait if we've reached max concurrent tasks
            while len(active_tasks) >= max_concurrent:
                active_tasks = [
                    task
                    for task in active_tasks
                    if task.status()["state"] in ("READY", "RUNNING")
                ]
                if len(active_tasks) >= max_concurrent:
                    time.sleep(10)

            task = export_image_to_asset(
                image=image,
                region=region,
                project_name=project_name,
                asset_folder=asset_folder,
                scale=scale,
                prefix=prefix,
            )
            task.start()
            active_tasks.append(task)
            all_tasks.append(task)
            print(f"Started export task {i+1} of {size}")

        # Monitor all tasks until completion
        results = monitor_tasks(all_tasks, timeout=timeout)

        # Add summary statistics
        results["total_tasks"] = size
        results["success_rate"] = len(results["completed"]) / size if size > 0 else 0

        return results

    except ee.EEException as e:
        raise ee.EEException(f"Collection export failed: {str(e)}")
