from rasterio.warp import transform as transform_coordinates
from config import GEOD

def getRasterResolution(raster) -> tuple[float, float]:
    if raster.crs is None:
        raise ValueError(f"{raster.name} has no CRS; ground resolution is unknown")

    col = (raster.width - 1) / 2
    row = (raster.height - 1) / 2
    points = [raster.transform * (col, row), raster.transform * (col + 1, row), raster.transform * (col, row + 1)]
    longitudes, latitudes = transform_coordinates(raster.crs, "EPSG:4326", [point[0] for point in points], [point[1] for point in points])
    _, _, x_m = GEOD.inv(longitudes[0], latitudes[0], longitudes[1], latitudes[1])
    _, _, y_m = GEOD.inv(longitudes[0], latitudes[0], longitudes[2], latitudes[2])
    return abs(float(x_m)), abs(float(y_m))

def getSamplingScales(raster, min_gsd) -> tuple[float, float, float]:
    x_gsd, y_gsd = getRasterResolution(raster)
    target_gsd = max(min_gsd, x_gsd, y_gsd)
    return target_gsd / x_gsd, target_gsd / y_gsd, target_gsd

def getWindowOrigins(total: int, span: float, step: float) -> list[float]:
    if total <= span:
        return [0.0]

    limit = total - span
    origins: list[float] = []
    origin = 0.0
    while origin <= limit + 1e-6:
        origins.append(origin)
        origin += step

    if not origins or abs(origins[-1] - limit) > 1e-6:
        origins.append(limit)

    return origins