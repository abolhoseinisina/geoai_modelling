from shapely.strtree import STRtree
from shapely.geometry import Polygon
from rasterio.transform import Affine
from shapely.validation import make_valid

from accuracy import getPolygonIoU

def convertPixelToPolygon(poly: Polygon, transform) -> Polygon:
    if not isinstance(transform, Affine):
        transform = Affine(*list(transform)[:6])
        
    return Polygon([transform * (px, py) for px, py in poly.exterior.coords])

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
    poly_map = convertPixelToPolygon(poly, transform)
    if not poly_map.is_valid:
        poly_map = make_valid(poly_map)
    
    if poly_map.geom_type == "MultiPolygon":
        poly_map = max(poly_map.geoms, key=lambda g: g.area)
    
    if poly_map.geom_type != "Polygon" or poly_map.is_empty:
        return None

    return poly_map

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
