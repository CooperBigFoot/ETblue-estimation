import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict

ee.Initialize()


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
            ee.Geometry.Point([lon, lat]),
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
    point_wgs84 = reproject_coordinates(x, y, from_epsg, to_epsg)
    lon, lat = point_wgs84.x, point_wgs84.y

    features = pandas_series_to_ee_features(daily_evapotranspiration, lon, lat)
    return ee.FeatureCollection(features).sort("system:time_start")


def export_feature_collection_to_asset(fc: ee.FeatureCollection, asset_id: str):
    """
    Export a FeatureCollection to Earth Engine Assets.

    Args:
        fc (ee.FeatureCollection): FeatureCollection to export.
        asset_id (str): Asset ID for the exported FeatureCollection.
    """
    task = ee.batch.Export.table.toAsset(
        collection=fc, description="Export daily evapotranspiration", assetId=asset_id
    )
    task.start()
    print(f"Export task started. Asset ID: {asset_id}")
