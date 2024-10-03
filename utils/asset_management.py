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


if __name__ == "__main__":
    # Example usage
    folder_path = (
        "projects/thurgau-irrigation/assets/Thurgau/ET_WaPOR_10m_dekadal_2018"
    )
    delete_folder_contents(folder_path)
