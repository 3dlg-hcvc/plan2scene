import math
import numpy as np

from arch_parser.models.room import Room
from arch_parser.models.wall_room_assignment import WallRoomAssignment


def ray_test(mask, ray_origin: tuple, ray_direction: tuple, ray_length: float) -> tuple:
    """
    Perform a ray-hit-test on the top-down view of the house geometry.
    :param mask: Top down view mask of walls
    :param ray_origin: Ray start point
    :param ray_direction: Ray direction
    :param ray_length: Maximum length of ray
    :return: Tuple (min_distance, hit point)
    """
    assert len(ray_origin) == 2
    assert len(ray_direction) == 2
    from shapely.geometry import GeometryCollection, LineString
    ray = [ray_origin, (ray_origin[0] + ray_direction[0] * ray_length, ray_origin[1] + ray_direction[1] * ray_length)]

    ray_segment = LineString(ray)
    ray_intersection = ray_segment.intersection(mask)

    intersection_point = None
    min_distance = ray_length

    hit_points = []
    if isinstance(ray_intersection, GeometryCollection):
        if not ray_intersection.is_empty:
            for g in ray_intersection.geoms:
                hit_points.extend([(r.x, r.y) for r in g.boundary])
    else:
        if not ray_intersection.is_empty:
            hit_points = [(r.x, r.y) for r in ray_intersection.boundary]

    for hit_point in hit_points:
        d = math.sqrt((hit_point[0] - ray_origin[0]) ** 2 + (hit_point[1] - ray_origin[1]) ** 2)
        if min_distance is None or d < min_distance:
            min_distance = d
            intersection_point = hit_point

    return min_distance, intersection_point


def get_room_mask(room: Room):
    """
    Compute top-down view wall mask of a room.
    :param room: Room considered
    :return Top down view wall mask as a shapely geometry:
    """
    from shapely.geometry import LineString
    processed_walls = []
    all_lines = []

    for wall_assignment in room.walls:
        assert isinstance(wall_assignment, WallRoomAssignment)
        wall = wall_assignment.wall

        if wall in processed_walls:
            continue

        ls = LineString([(wall.p1[0], wall.p1[2]), (wall.p2[0], wall.p2[2])]).buffer(1)

        for hole in wall.holes:
            h_start_x = ((wall.p2[0] - wall.p1[0]) * hole.start) + wall.p1[0]
            h_end_x = ((wall.p2[0] - wall.p1[0]) * hole.end) + wall.p1[0]
            h_start_y = ((wall.p2[2] - wall.p1[2]) * hole.start) + wall.p1[2]
            h_end_y = ((wall.p2[2] - wall.p1[2]) * hole.end) + wall.p1[2]

            hs = LineString([(h_start_x, h_start_y), (h_end_x, h_end_y)]).buffer(1)
            ls = ls.difference(hs)

        all_lines.append(ls)

    from shapely.ops import cascaded_union
    u = cascaded_union(all_lines)
    return u


def get_transform(x: float, y: float, z: float, angle: float, scale_x: float = 1, scale_z: float = 1) -> list:
    """
    Compute transformation matrix
    :param x: X position of center
    :param y: Y position of center
    :param z: Z position of center
    :param angle: Rotation of object about Y axis, in degrees.
    :param scale_x: Scaling factor along X axis
    :param scale_z: Scaling factor along Z axis
    :return: Transformation matrix as a nested list
    """
    from scipy.spatial.transform import Rotation as R
    scale_mat = np.array([[scale_x, 0, 0, 0], [0, 1, 0, 0], [0, 0, scale_z, 0], [0, 0, 0, 1]], dtype=float)
    r = R.from_rotvec((0, angle / 180 * np.pi, 0))
    m = r.as_dcm()
    m = np.vstack([m, [x, y, z]])
    m = np.hstack([m, np.transpose(np.array([[0, 0, 0, 1]]))])
    m = np.matmul(scale_mat, m)
    return m.flatten().tolist()


def find_angle(p: tuple) -> float:
    """
    Computes angle of a 2D vector.
    Angles are measured with respect to positive x axis, on the counter clockwise direction. Negative y is considered up. Positive x is considered right.
    :param p: Vector considered in the form (x, y)
    :return: Angle in degrees.
    """
    assert len(p) == 2
    h = math.sqrt(p[0] ** 2 + p[1] ** 2)
    ang = math.acos(p[0] / h) * 180 / math.pi
    if p[1] > 0:
        ang = -ang
    return ang


def l2_dist_sq(p1: tuple, p2: tuple) -> float:
    """
    Compute squared L2 distance between p1 and p2
    :param p1: Point 1 in the form (x, y)
    :param p2: Point 2 in the form (x, y)
    :return: Squared L2 distance
    """
    assert len(p1) == 2
    assert len(p2) == 2
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx ** 2 + dy ** 2


def hole_to_line(wall_p1: tuple, wall_p2: tuple, hole_start: float, hole_end: float) -> tuple:
    """
    Converts a 2D hole definition to a line segment.
    :param wall_p1: Top-down x,y coordinate of wall start point
    :param wall_p2: Top-down x,y coordinate of wall end point
    :param hole_start: Distance from wall_p1 to start of hole
    :param hole_end: Distance from wall_p1 to end of hole
    :return: Line segment as tuple(tuple(x1,y1), tuple(x2,y2))
    """
    assert len(wall_p1) == 2
    assert len(wall_p2) == 2

    wall_length = math.sqrt(l2_dist_sq(wall_p1, wall_p2))
    hole_start = hole_start / wall_length
    hole_end = hole_end / wall_length
    start_x = (wall_p2[0] - wall_p1[0]) * hole_start + wall_p1[0]
    start_y = (wall_p2[1] - wall_p1[1]) * hole_start + wall_p1[1]
    end_x = (wall_p2[0] - wall_p1[0]) * hole_end + wall_p1[0]
    end_y = (wall_p2[1] - wall_p1[1]) * hole_end + wall_p1[1]
    return (start_x, start_y), (end_x, end_y)


def point_line_seg_distance(p1: tuple, p2: tuple, p: tuple, extend_line: bool = False) -> tuple:
    """
    Compute distance from 2D point p to line segment (p1, p2).
    :param p1: Start point of line in the form (x, y)
    :param p2: End point of line in the form (x, y)
    :param p: Point to compute distance to, in the form (x, y)
    :param extend_line: Should the line be extended as necessary to reduce the distance.
    :return: Tuple(Distance, (x,y) of closest point)
    """
    # Code adapted from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    assert len(p1) == 2
    assert len(p2) == 2
    assert len(p) == 2

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p
    x1, x2, x3, y1, y2, y3 = float(x1), float(x2), float(x3), float(y1), float(y2), float(y3)
    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py
    if norm == 0:
        dx = x1 - x3
        dy = y1 - y3

        dist = (dx * dx + dy * dy) ** .5
        return dist, (x1, y1)

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if not extend_line:
        if u > 1:
            u = 1
        elif u < 0:
            u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = (dx * dx + dy * dy) ** .5

    return dist, (x, y)
