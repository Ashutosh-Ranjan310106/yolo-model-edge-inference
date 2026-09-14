"""
Geometry Utilities for Spatial Navigation.
Implements 2D polygon intersection (Sutherland-Hodgman) and area calculations.
"""

from typing import List, Tuple
import numpy as np

def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Computes area of a 2D polygon using the Shoelace formula."""
    if len(points) < 3:
        return 0.0
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def clip_polygon_with_line(
    polygon: List[Tuple[float, float]],
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Clips a polygon against an infinite directed line (Sutherland-Hodgman step)."""
    def is_inside(p: Tuple[float, float]) -> bool:
        # Cross product to determine which side of the line point p lies on
        return (line_p2[0] - line_p1[0]) * (p[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (p[0] - line_p1[0]) >= 0

    def intersection(cp1: Tuple[float, float], cp2: Tuple[float, float]) -> Tuple[float, float]:
        dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
        dp = (line_p1[0] - line_p2[0], line_p1[1] - line_p2[1])
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = line_p1[0] * line_p2[1] - line_p1[1] * line_p2[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0] + 1e-10)
        return ((n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3)

    output_list = []
    if not polygon:
        return output_list

    s = polygon[-1]
    for e in polygon:
        if is_inside(e):
            if not is_inside(s):
                output_list.append(intersection(s, e))
            output_list.append(e)
        elif is_inside(s):
            output_list.append(intersection(s, e))
        s = e
    return output_list

def polygon_intersection(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    """Calculates intersection area between two convex polygons."""
    clipped = list(poly1)
    for i in range(len(poly2)):
        p1 = poly2[i]
        p2 = poly2[(i + 1) % len(poly2)]
        clipped = clip_polygon_with_line(clipped, p1, p2)
        if not clipped:
            return 0.0
    return polygon_area(clipped)

def box_to_polygon(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float]]:
    """Converts bounding box [x1, y1, x2, y2] to clockwise polygon points."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
