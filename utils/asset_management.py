import ee


def delete_folder_contents(folder_path):
    # Initialize the Earth Engine API
    ee.Initialize(project="thurgau-irrigation")

    # List all assets in the specified folder
    assets = ee.data.listAssets({"parent": folder_path})

    # Extract the asset IDs
    asset_ids = [asset["id"] for asset in assets["assets"]]

    # Delete each asset
    for asset_id in asset_ids:
        try:
            ee.data.deleteAsset(asset_id)
            print(f"Deleted: {asset_id}")
        except ee.EEException as e:
            print(f"Error deleting {asset_id}: {str(e)}")


def keep_first_n_images(collection_id: str, n: int) -> str:
    """
    Keeps the first n images in a GEE image collection and deletes the rest.

    Args:
        collection_id (str): The asset ID of the image collection.
        n (int): The number of images to keep.

    Returns:
        str: A message indicating the result of the operation.
    """
    try:

        ee.Initialize(project="thurgau-irrigation")

        collection = ee.ImageCollection(collection_id)

        initial_size = collection.size().getInfo()

        if n >= initial_size:
            return f"No images deleted. Collection size ({initial_size}) is less than or equal to {n}."

        image_ids = collection.aggregate_array("system:id").getInfo()

        images_to_delete = image_ids[n:]

        for image_id in images_to_delete:
            ee.data.deleteAsset(image_id)

        final_size = ee.ImageCollection(collection_id).size().getInfo()

        return (
            f"Operation successful. Initial collection size: {initial_size}, "
            f"Final collection size: {final_size}, Images deleted: {initial_size - final_size}"
        )

    except ee.EEException as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":

    collection_id = "projects/thurgau-irrigation/assets/Thurgau/ET_blue_postprocessed_wapor_dekadal_refactored_2018-2023"
    delete_folder_contents(collection_id)

    # # Example usage of keep_first_n_images
    # collection_id = "projects/thurgau-irrigation/assets/Thurgau/ET_blue_raw_2018-2022"
    # result = keep_first_n_images(collection_id, 108)
    # print(result)
