from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from rasterio.transform import Affine
from shapely.validation import make_valid

from accuracy import getPolygonIoU

def convertPixelToPolygon(poly: Polygon, transform) -> Polygon:
    if not isinstance(transform, Affine):
        transform = Affine(*list(transform)[:6])
        
    return Polygon([transform * (px, py) for px, py in poly.exterior.coords])

def repairPolygon(poly):
    if poly is None or poly.is_empty:
        return None

    if not poly.is_valid:
        poly = make_valid(poly)

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda part: part.area)

    if poly.geom_type != "Polygon" or poly.is_empty:
        return None

    return poly

def mergeOverlappingPredictions(polygons: list[Polygon]) -> list[Polygon]:
    cleaned = [poly for poly in (repairPolygon(poly) for poly in polygons) if poly is not None]
    if not cleaned:
        return []

    merged = unary_union(cleaned)
    if merged.is_empty:
        return []
    if merged.geom_type == "Polygon":
        parts = [merged]
    elif merged.geom_type == "MultiPolygon":
        parts = list(merged.geoms)
    else:
        parts = list(getattr(merged, "geoms", []))

    out = []
    for part in parts:
        repaired = repairPolygon(part)
        if repaired is not None:
            out.append(repaired)
    
    return out

def performNMS(polygons: list[Polygon], scores: list[float], iou_thresh: float) -> list[int]:
    if not polygons:
        return []

    order = sorted(range(len(polygons)), key=lambda i: scores[i], reverse=True)
    tree = STRtree(polygons)
    kept: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue

        kept.append(i)
        for j in tree.query(polygons[i]):
            j = int(j)
            if j == i or j in suppressed:
                continue
            
            if scores[j] > scores[i]:
                continue
            
            if getPolygonIoU(polygons[i], polygons[j]) >= iou_thresh:
                suppressed.add(j)

    return kept

def georeferencePolygon(poly: Polygon, transform):
    return repairPolygon(convertPixelToPolygon(poly, transform))

def georeferencePolygons(pixel_polys: list[Polygon], scores: list[float], transform) -> tuple[list[Polygon], list[float]]:
    mapped: list[Polygon] = []
    mapped_scores: list[float] = []
    for pixel_poly, score in zip(pixel_polys, scores):
        poly_map = georeferencePolygon(pixel_poly, transform)
        if poly_map == None:
            continue
        
        mapped.append(poly_map)
        mapped_scores.append(float(score))
    
    return mapped, mapped_scores
