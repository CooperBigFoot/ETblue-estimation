import ee
import pandas as pd
from datetime import datetime


def print_collection_dates(collection: ee.ImageCollection) -> None:
    """
    Print the dates of all images in an ImageCollection.

    Args:
        collection (ee.ImageCollection): The input image collection.

    Returns:
        None: This function prints the dates to the console.
    """
    # Get a list of all image dates
    dates = collection.aggregate_array("system:time_start")

    # Convert to ee.Date objects and format as strings
    formatted_dates = dates.map(lambda d: ee.Date(d).format("YYYY-MM-dd"))

    # Get the list of formatted dates
    date_list = formatted_dates.getInfo()

    print("Dates of images in the collection:")
    for date in date_list:
        print(date)


def store_collection_dates(collection: ee.ImageCollection) -> pd.DataFrame:
    """
    Store the dates of all images in an ImageCollection in a pandas DataFrame.

    Args:
        collection (ee.ImageCollection): The input image collection.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the dates in datetime format.
    """
    dates = collection.aggregate_array("system:time_start")
    formatted_dates = dates.map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
    date_list = formatted_dates.getInfo()

    date_df = pd.DataFrame({"date": pd.to_datetime(date_list)})

    return date_df


def update_image_timestamp(
    collection: ee.ImageCollection, image_id: str, date_str: str
) -> ee.ImageCollection:
    """
    Update the 'system:time_start' for a specific image in the collection.

    Args:
        collection (ee.ImageCollection): The original image collection.
        image_id (str): The ID of the image to update.
        date_str (str): The date string in 'YYYY-MM-DD' format.

    Returns:
        ee.ImageCollection: Updated image collection.
    """
    # Convert the date string to a timestamp
    date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(date.timestamp() * 1000)  # Convert to milliseconds

    # Function to update the image if it matches the ID
    def update_image(image):
        return ee.Algorithms.If(
            ee.String(image.get("system:index")).equals(image_id),
            image.set("system:time_start", timestamp),
            image,
        )

    # Map the update function over the collection
    updated_collection = collection.map(update_image)

    return updated_collection