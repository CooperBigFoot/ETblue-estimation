import ee


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
