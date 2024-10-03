import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def reproject_coordinates(
    x: float, y: float, from_epsg: int = 2056, to_epsg: int = 4326
) -> Point:
    """
    Reproject coordinates from one coordinate system to another.

    Args:
        x (float): X-coordinate (easting) in the source coordinate system.
        y (float): Y-coordinate (northing) in the source coordinate system.
        from_epsg (int): EPSG code of the source coordinate system. Default is 2056 (CH1903+).
        to_epsg (int): EPSG code of the target coordinate system. Default is 4326 (WGS84).

    Returns:
        Point: Shapely Point object with reprojected coordinates.
    """
    point = gpd.GeoSeries([Point(x, y)], crs=f"EPSG:{from_epsg}")
    point_reprojected = point.to_crs(f"EPSG:{to_epsg}")
    return point_reprojected[0]


def pandas_series_to_ee_features(
    series: pd.Series, lon: float, lat: float
) -> List[ee.Feature]:
    """
    Convert a pandas Series to a list of Earth Engine Features.

    Args:
        series (pd.Series): Input pandas Series with DatetimeIndex and float values.
        lon (float): Longitude of the station in WGS84.
        lat (float): Latitude of the station in WGS84.

    Returns:
        List[ee.Feature]: List of Earth Engine Features.
    """
    features = []
    for date, value in series.items():
        feature = ee.Feature(
            ee.Geometry.Point([lon, lat]).transform("EPSG:4326"),
            {
                "date_[YYYYmmdd]": date.strftime("%Y-%m-%d"),
                "evapotranspiration_[mm/d]": float(value),
                "system:time_start": ee.Date(date.strftime("%Y-%m-%d")).millis(),
            },
        )
        features.append(feature)
    return features


def create_ee_feature_collection(
    daily_evapotranspiration: pd.Series,
    x: float,
    y: float,
    from_epsg: int = 2056,
    to_epsg: int = 4326,
) -> ee.FeatureCollection:
    """
    Create an Earth Engine FeatureCollection from a pandas Series and station coordinates.

    Args:
        daily_evapotranspiration (pd.Series): Input pandas Series with DatetimeIndex and float values.
        x (float): X-coordinate (easting) of the station in CH1903+ (EPSG:2056).
        y (float): Y-coordinate (northing) of the station in CH1903+ (EPSG:2056).
        from_epsg (int): EPSG code of the source coordinate system. Default is 2056 (CH1903+).
        to_epsg (int): EPSG code of the target coordinate system. Default is 4326 (WGS84).

    Returns:
        ee.FeatureCollection: Earth Engine FeatureCollection containing the time series data.
    """
    point_wgs84 = Point([x, y])
    lon, lat = point_wgs84.xy[0][0], point_wgs84.xy[1][0]

    features = pandas_series_to_ee_features(daily_evapotranspiration, lon, lat)
    return ee.FeatureCollection(features).sort("system:time_start")


def extract_raster_values(
    raster_image: ee.Image, feature_collection: ee.FeatureCollection
) -> ee.FeatureCollection:
    """
    Extract raster values at the locations specified by the feature collection.

    Args:
        raster_image (ee.Image): Single-band raster image with evapotranspiration values.
        feature_collection (ee.FeatureCollection): Collection of flux net station measurements.

    Returns:
        ee.FeatureCollection: Original collection with added raster value.
    """

    def extract_at_point(feature: ee.Feature) -> ee.Feature:
        point_value = raster_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=10,  # 10m resolution as specified
        ).get("downscaled")
        return feature.set("raster_et", point_value)

    return feature_collection.map(extract_at_point)


def prepare_data_for_plotting(feature_collection: ee.FeatureCollection) -> pd.DataFrame:
    """
    Prepare the feature collection data for plotting by converting to a pandas DataFrame.

    Args:
        feature_collection (ee.FeatureCollection): Collection with flux net and raster values.

    Returns:
        pd.DataFrame: DataFrame ready for plotting.
    """
    # Get the data as a list of dictionaries
    data = feature_collection.getInfo()["features"]

    # Convert to pandas DataFrame
    df = pd.DataFrame(
        [
            {
                "date": datetime.strptime(
                    feat["properties"]["date_[YYYYmmdd]"], "%Y-%m-%d"
                ),
                "flux_net_et": feat["properties"]["evapotranspiration_[mm/month]"],
                "raster_et": feat["properties"].get(
                    "raster_et", None
                ),  # Use .get() to avoid KeyError
            }
            for feat in data
        ]
    )

    # Ensure the date is set as the index and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df
