import json
import random
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
import geopandas as gpd
from pathlib import Path
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.validation import make_valid
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform
from rasterio.windows import bounds as window_bounds
from geo import BUILDINGS_GEOJSON, DATA_DIR, TILE_SIZE, getSamplingScales, getWindowOrigins

SEED = 42
OVERLAP = 128
NEGATIVE_RATIO = 0.5
IMAGE_NAME_COLUMN = "image_name"
MAX_BLANK_FRACTION = 0.5
MIN_INSTANCE_AREA_PX = 24

def getMaskOutline(mask: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(mask, dtype=bool)
    edges[1:, :] |= mask[1:, :] ^ mask[:-1, :]
    edges[:, 1:] |= mask[:, 1:] ^ mask[:, :-1]
    return edges

def writeAlignmentCheck(tiles_dir: Path, record: dict, output_path: Path) -> None:
    image = np.array(Image.open(tiles_dir / record["image"]).convert("RGB"))
    mask = np.zeros(image.shape[:2], dtype=bool)
    for rings in record["polygons"]:
        polygon = Polygon(rings[0], rings[1:])
        mask |= rasterize([(polygon, 1)], out_shape=image.shape[:2], dtype="uint8").astype(bool)

    overlay = image.copy()
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
    overlay[getMaskOutline(mask)] = [0, 255, 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(f'Check "{output_path.name}" ({record["image"]}, {len(record["polygons"])} buildings): the green outlines should sit on roof edges.')

def loadTrainingBuildings() -> gpd.GeoDataFrame:
    if not BUILDINGS_GEOJSON.exists():
        raise SystemExit(f"missing footprints: {BUILDINGS_GEOJSON}")

    footprints = gpd.read_file(BUILDINGS_GEOJSON)
    if footprints.crs is None:
        raise SystemExit(f"{BUILDINGS_GEOJSON.name} has no CRS. Set one explicitly before tiling, otherwise the footprints cannot be placed on the imagery.")

    if IMAGE_NAME_COLUMN not in footprints.columns:
        raise SystemExit(f"{BUILDINGS_GEOJSON.name} is missing the '{IMAGE_NAME_COLUMN}' column that maps each building to its source image.")

    footprints = footprints[footprints.geometry.notna() & ~footprints.geometry.is_empty].copy()
    footprints[IMAGE_NAME_COLUMN] = footprints[IMAGE_NAME_COLUMN].astype(str).str.strip()
    footprints = footprints[footprints[IMAGE_NAME_COLUMN].ne("") & footprints[IMAGE_NAME_COLUMN].ne("None")]
    if footprints.empty:
        raise SystemExit(f"{BUILDINGS_GEOJSON.name} has no usable footprint features.")

    return footprints

def getImagePath(image_name: str):
    image_path = DATA_DIR / f"{image_name}.tif"
    if image_path.exists():
        return image_path
    
    return None

def pairImagesWithFootprints(buildings: gpd.GeoDataFrame) -> list[tuple]:
    pairs: list[tuple] = []
    missing: list[str] = []
    for image_name, subset in buildings.groupby(IMAGE_NAME_COLUMN, sort=True):
        tif_path = getImagePath(image_name)
        if tif_path is None:
            missing.append(image_name)
            continue
        pairs.append((tif_path, subset.reset_index(drop=True)))

    if missing:
        print(f"warning: {len(missing)} image_name value(s) have no matching .tif under {DATA_DIR.name}/:")
        for name in missing:
            print(f"  - {name}")

    if not pairs:
        raise SystemExit(f"No imagery matched the '{IMAGE_NAME_COLUMN}' values in {BUILDINGS_GEOJSON.name}. Expected files like {DATA_DIR / '<image_name>.tif'}.")

    return pairs

def transformFootprints(gdf: gpd.GeoDataFrame, raster) -> gpd.GeoDataFrame:
    if gdf.crs != raster.crs:
        gdf = gdf.to_crs(raster.crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"].reset_index(drop=True)
    if gdf.empty:
        raise SystemExit(f"No valid polygons remain for {raster.name} after cleanup.")

    raster_box = box(*raster.bounds)
    overlap = box(*gdf.total_bounds).intersection(raster_box).area
    if overlap == 0:
        raise SystemExit(
            f"Footprints for {raster.name} do not overlap the imagery after reprojection.\n"
            f"  footprints: {tuple(round(v, 2) for v in gdf.total_bounds)}\n"
            f"  imagery:    {tuple(round(v, 2) for v in raster.bounds)}\n"
            "The declared CRS of one of the two files is probably wrong."
        )

    return gdf

def getRGBStretchBounds(raster, band_indexes) -> tuple[np.ndarray, np.ndarray] | None:
    if raster.dtypes[0] == "uint8":
        return None

    decimated = raster.read(band_indexes, out_shape=(len(band_indexes), min(raster.height, 1024), min(raster.width, 1024)))
    flat = decimated.reshape(len(band_indexes), -1)
    return np.percentile(flat, 2, axis=1), np.percentile(flat, 98, axis=1)

def getRingCoords(ring) -> list[list[float]]:
    return [[round(x, 1), round(y, 1)] for x, y in ring.coords]

def transformPolygonToPixelCoordinates(geom, transform: Affine):
    inverse = ~transform
    matrix = (inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f)
    return affine_transform(geom, matrix)

def tilePolygons(gdf, spatial_index, tile_geom, tile_transform, size) -> list[list]:
    pixel_box = box(0, 0, size, size)
    polygons = []
    for polygon_index in spatial_index.query(tile_geom, predicate="intersects"):
        clipped = transformPolygonToPixelCoordinates(gdf.geometry.iloc[polygon_index], tile_transform)
        clipped = clipped.intersection(pixel_box)
        if clipped.is_empty:
            continue

        parts = [clipped] if isinstance(clipped, Polygon) else list(getattr(clipped, "geoms", []))
        for part in parts:
            if not isinstance(part, Polygon) or part.area < MIN_INSTANCE_AREA_PX:
                continue
            polygons.append([getRingCoords(part.exterior)] + [getRingCoords(r) for r in part.interiors])
    return polygons

def stretch2uint8(array: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scaled = (array.astype(np.float32) - low[:, None, None]) / np.maximum((high - low)[:, None, None], 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

def readImageTile(src, window, band_indexes, stretch, tile_size: int = TILE_SIZE) -> np.ndarray | None:
    array = src.read(band_indexes, window=window, out_shape=(3, tile_size, tile_size), resampling=Resampling.bilinear, boundless=True, fill_value=0)
    if stretch is not None:
        array = stretch2uint8(array, *stretch)

    if (array.max(axis=0) == 0).mean() > MAX_BLANK_FRACTION:
        return None

    return array

def writeImageTile(tiles_dir: Path, tif_path, array, transform, crs, polygons, records: list[dict], x0, y0, gsd_m, tile_size: int = TILE_SIZE) -> None:
    name = f"{tif_path.stem}_{round(x0):05d}_{round(y0):05d}.png"
    Image.fromarray(np.moveaxis(array, 0, -1)).save(tiles_dir / "images" / name)
    records.append(
        {
            "image": f"images/{name}",
            "source": tif_path.name,
            "size": tile_size,
            "crs": crs,
            "transform": list(transform)[:6],
            "gsd_m": gsd_m,
            "polygons": polygons,
        }
    )

def tileImage(tiles_dir: Path, tif_path, footprints: gpd.GeoDataFrame, records: list[dict], tile_size: int = TILE_SIZE) -> None:
    with rasterio.open(tif_path) as src:
        scale_x, scale_y, target_gsd = getSamplingScales(src)
        footprints = transformFootprints(footprints, src)
        band_indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        stretch = getRGBStretchBounds(src, band_indexes)
        spatial_index = footprints.sindex

        source_width = tile_size * scale_x
        source_height = tile_size * scale_y
        step_x = (tile_size - OVERLAP) * scale_x
        step_y = (tile_size - OVERLAP) * scale_y

        negatives: list[tuple] = []
        positives_before = len(records)
        for y0 in getWindowOrigins(src.height, source_height, step_y):
            for x0 in getWindowOrigins(src.width, source_width, step_x):
                window = Window(x0, y0, source_width, source_height)
                transform = src.window_transform(window) * Affine.scale(scale_x, scale_y)
                tile_geom = box(*window_bounds(window, src.transform))
                polygons = tilePolygons(footprints, spatial_index, tile_geom, transform, tile_size)
                if not polygons:
                    negatives.append((x0, y0, window, transform))
                    continue

                array = readImageTile(src, window, band_indexes, stretch, tile_size)
                if array is None:
                    continue

                writeImageTile(tiles_dir, tif_path, array, transform, src.crs.to_string(), polygons, records, x0, y0, target_gsd, tile_size)

        n_positive = len(records) - positives_before
        rng = random.Random(SEED)
        rng.shuffle(negatives)
        quota = min(len(negatives), max(1, int(round(n_positive * NEGATIVE_RATIO))))
        for x0, y0, window, transform in negatives[:quota]:
            array = readImageTile(src, window, band_indexes, stretch, tile_size)
            if array is None:
                continue
            writeImageTile(tiles_dir, tif_path, array, transform, src.crs.to_string(), [], records, x0, y0, target_gsd, tile_size)

def executeTiling(tiles_dir: Path, tile_index: Path, alignment_path: Path | None = None) -> list[dict]:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"missing imagery directory: {DATA_DIR}")

    buildings = loadTrainingBuildings()
    tif_building_pairs = pairImagesWithFootprints(buildings)
    print(f"Training data: {BUILDINGS_GEOJSON.name} ({len(buildings)} buildings -> {len(tif_building_pairs)} images)")

    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    (tiles_dir / "images").mkdir(parents=True)

    records: list[dict] = []
    for tif_path, footprints in tqdm(tif_building_pairs, desc="Tiling training dataset images", ncols=100):
        tileImage(tiles_dir, tif_path, footprints, records)

    if not records:
        raise SystemExit("No tiles were written - check the alignment warnings above.")

    tile_index.write_text(json.dumps(records))
    positives = sum(1 for record in records if record["polygons"])
    negatives = len(records) - positives
    instances = sum(len(record["polygons"]) for record in records)
    print(f"\nWrote {len(records)} tiles ({positives} with buildings, {negatives} empty) and {instances} building instances to {tiles_dir.name}")

    if alignment_path is not None:
        labeled = [record for record in records if record["polygons"]]
        writeAlignmentCheck(tiles_dir, random.choice(labeled), alignment_path)

    return records