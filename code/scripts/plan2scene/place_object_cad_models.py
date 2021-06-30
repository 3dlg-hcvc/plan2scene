#!/bin/python
import logging

from shapely.geometry import Polygon

from arch_parser.models.house import House
from arch_parser.models.object import ObjectAnnotation
from arch_parser.models.wall import Wall
from arch_parser.parser import parse_house_json_file
from arch_parser.serializer import serialize_scene_json
from plan2scene.house_gen.geom_util import hole_to_line, point_line_seg_distance, find_angle, get_transform, get_room_mask, ray_test
from config_parser import Config
from plan2scene.config_manager import ConfigManager
import os
import os.path as osp
import json
import numpy as np
import math

LARGE_VALUE = 10000000000000


def compute_object_rotation(p1, p2, ratio: str, face_hole: bool, room_walls: list, full_wall_mask) -> float:
    """
    Compute YAW rotation angle of an object.
    :param p1: Position AABB start point
    :param p2: Position AABB end point
    :param ratio: wide or narrow. Wide object examples: TVs, Cabinets. Narrow object example: Toilets.
    :param face_hole: Should the object face the wall with a door?
    :param room_walls: Walls of the room
    :param full_wall_mask: Mask of the room boundary
    :return: Angle in degrees
    """
    from shapely.ops import polygonize
    assert ratio in ["wide", "narrow"]
    assert len(p1) == 2
    assert len(p2) == 2

    dx = abs(p2[0] - p1[0])
    dz = abs(p2[1] - p1[1])
    p = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    wall_outlines = []
    hole_walls = []
    for wall in room_walls:
        if wall.holes:
            hole_walls.append(wall)
        wall_outlines.append((wall.p1, wall.p2))

    if face_hole and hole_walls:
        # Old heuristic method for closets. No longer used.
        hole_wall = hole_walls[0]

        wall_polygon = list(polygonize(wall_outlines))[0]
        room_center = wall_polygon.centroid.coords[:][0]
        _, closest_point = point_line_seg_distance((hole_wall.p1[0], hole_wall.p1[2]), (hole_wall.p2[0], hole_wall.p2[2]), room_center, extend_line=True)
        degree = (np.degrees(math.atan2(closest_point[1] - room_center[1], closest_point[0] - room_center[0])) - 90) % 360
        return degree
    else:
        if ratio == "wide":
            # The object is wide. (Length parallel to the wall facing side is greater than the length perpendicular to wall facing side)
            # E.g. Televisions, cupboards
            if dx > dz:
                d_plus, _ = ray_test(full_wall_mask, p, (0, 1), 10000)
                d_minus, _ = ray_test(full_wall_mask, p, (0, -1), 10000)
                if d_plus > d_minus:
                    return 0
                else:
                    return 180
            else:
                d_plus, _ = ray_test(full_wall_mask, p, (1, 0), 10000)
                d_minus, _ = ray_test(full_wall_mask, p, (-1, 0), 10000)
                if d_plus > d_minus:
                    return 270
                else:
                    return 90
        elif ratio == "narrow":
            # The object is narrow. (Length parallel to the wall facing side is shorter than the length perpendicular to wall facing side)
            # E.g. Toilet
            if dx > dz:  # TODO: Need to verify this case
                d_plus, _ = ray_test(full_wall_mask, p, (1, 0), 10000)
                d_minus, _ = ray_test(full_wall_mask, p, (-1, 0), 10000)
                if d_plus > d_minus:
                    return 270
                else:
                    return 90
            else:
                d_plus, _ = ray_test(full_wall_mask, p, (0, 1), 10000)
                d_minus, _ = ray_test(full_wall_mask, p, (0, -1), 10000)
                if d_plus > d_minus:
                    return 0
                else:
                    return 180
                # return 0 #or 180
    assert False


def assign_objects_to_rooms(rooms: dict, objects: list, arch_defaults: Config) -> dict:
    """
    Assign objects to rooms.
    :param rooms: Dictionary of rooms of a house
    :param objects: List of ObjectAnnotations
    :param arch_defaults: conf.house_gen
    :return: Dictionary mapping from room_id to a list of objects assigned to the room.
    """
    room_key_obj = []
    room_key_objects_map = {}
    for room_key, room in rooms.items():
        # room.objects = []
        room_key_objects_map[room_key] = []
        polyline = room.get_polyline()
        room_polygon = Polygon([(a[0], a[2]) for a in polyline])
        for obj in objects:
            obj_box = Polygon([(obj.p1[0], obj.p1[1]), (obj.p2[0], obj.p1[1]), (obj.p2[0], obj.p2[1]), (obj.p1[0], obj.p2[1])])
            if obj_box.intersection(room_polygon).area > obj_box.area * arch_defaults.object_aabb_room_intersection_threshold:
                room_key_obj.append((obj_box.intersection(room_polygon).area / obj_box.area, room_key, obj))

    room_key_obj = sorted(room_key_obj, key=lambda a: a[0], reverse=True)
    assigned_objects = []
    for overlap, room_key, obj in room_key_obj:
        if obj not in assigned_objects:
            room = rooms[room_key]
            room_key_objects_map[room_key].append(obj)
            assigned_objects.append(obj)
            # logging.info("Assigned: %s => %s" % (str(obj), str(room.types)))

    return room_key_objects_map


def find_closest_wall(candidate_points, candidate_walls, house_gen_conf: Config) -> Wall:
    """
    Find closest wall to candidate points
    :param candidate_points: Candidate points
    :param candidate_walls: Candidate walls
    :param house_gen_conf: house_gen_conf
    :return: 
    """
    shortest_distance = LARGE_VALUE
    closest_wall = None
    for p in candidate_points:
        for wall in candidate_walls:
            dist, c_point = point_line_seg_distance((wall.p1[0], wall.p1[2]), (wall.p2[0], wall.p2[2]), p)
            # c_point should not lay in a hole
            in_hole = False
            for hole in wall.holes:
                if hole.hole_type != "Door":
                    continue
                hole_start, hole_end = hole_to_line((wall.p1[0], wall.p1[2]), (wall.p2[0], wall.p2[2]), hole.start, hole.end)
                if point_line_seg_distance(hole_start, hole_end, c_point)[0] <= house_gen_conf.in_hole_distance_threshold:
                    in_hole = True
                    break
            if not in_hole and dist < shortest_distance:
                shortest_distance = dist
                closest_wall = wall

    return closest_wall


def get_object_json(obj: ObjectAnnotation, house: House, contained_room_id: str, house_gen_conf: Config, index: int) -> tuple:
    """
    Generate Scene JSON representation of an object.
    :param obj: Object Annotation
    :param house: House containing the object
    :param contained_room_id: Room id of the room containing obj
    :param house_gen_conf: house_gen_conf from config
    :param index: Index of object
    :return: tuple (scene json entry for the object, index of next object)
    """
    # logging.info("Processing object: {object}".format(object=obj.type))
    room_walls = [a.wall for a in house.rooms[contained_room_id].walls]

    p = (obj.p1[0] + obj.p2[0]) / 2.0, (obj.p1[1] + obj.p2[1]) / 2.0
    dx = abs(obj.p1[0] - obj.p2[0])
    dz = abs(obj.p1[1] - obj.p2[1])
    scale_x = 1.0
    scale_z = 1.0

    if obj.type in house_gen_conf.object_type_specific_rules.use_centroid_as_corners:
        corner_points = [((obj.p1[0] + obj.p2[0]) / 2, (obj.p1[1] + obj.p2[1]) / 2)]
    else:
        corner_points = [(obj.p1[0], obj.p1[1]), (obj.p2[0], obj.p1[1]), (obj.p2[0], obj.p2[1]), (obj.p1[0], obj.p2[1])]

    # Find closest wall
    closest_wall = find_closest_wall(corner_points, room_walls, house_gen_conf)

    # Calculate shortest_distance and closest point using the center of annotation
    shortest_distance = None
    closest_point = None
    available_wall_length = None
    closest_wall_length = None
    if closest_wall is not None:
        shortest_distance, closest_point = point_line_seg_distance((closest_wall.p1[0], closest_wall.p1[2]), (closest_wall.p2[0], closest_wall.p2[2]), p,
                                                                   extend_line=True)

        # Distance to corners of the closest wall from the projection of object center to the wall can be a factor in determining CAD model
        dist_to_corner1 = ((closest_point[0] - closest_wall.p1[0]) ** 2 + (closest_point[1] - closest_wall.p1[2]) ** 2) ** 0.5
        dist_to_corner2 = ((closest_point[0] - closest_wall.p2[0]) ** 2 + (closest_point[1] - closest_wall.p2[2]) ** 2) ** 0.5
        available_wall_length = min(dist_to_corner1, dist_to_corner2) * 2
        closest_wall_length = ((closest_wall.p1[0] - closest_wall.p2[0]) ** 2 + (closest_wall.p1[2] - closest_wall.p2[2]) ** 2) ** 0.5

    # Size of the largest hole in room can be a criteria in determining model
    hole_sizes = []
    for wall in room_walls:
        wall_length = ((wall.p2[0] - wall.p1[0]) ** 2 + (wall.p2[2] - wall.p1[2]) ** 2) ** 0.5
        for hole in wall.holes:
            hole_sizes.append(abs(hole.end - hole.start) * wall_length)
    hole_sizes = sorted(hole_sizes, reverse=True)

    # Room types is a criteria in determining suitable CAD model
    room_types = house.rooms[contained_room_id].types

    # Obtain the most suitable CAD model for the object
    object_model = find_suitable_cad_model(obj, closest_wall=closest_wall, house=house, house_gen_conf=house_gen_conf, room_types=room_types,
                                           room_walls=room_walls,
                                           hole_sizes=hole_sizes, available_wall_length=available_wall_length, max_box_length=max(dx, dz))
    if object_model is None:
        logging.error("No model for object type: {type}".format(type=obj.type))
        if obj.type != "shower":
            assert False
        return None, index

    lift = 0  # Vertical displacement
    if "lift" in object_model:
        lift = object_model["lift"]

    if "clamp_wall" in object_model and object_model["clamp_wall"] and closest_point is not None and shortest_distance is not None:
        # The CAD model is clamped to the closest wall
        correction_direction = ((closest_point[0] - p[0]) / shortest_distance, (closest_point[1] - p[1]) / shortest_distance)
        clamp_clearance = object_model["clamp_clearance"] + house_gen_conf.def_wall_depth / 2
        movement = shortest_distance - clamp_clearance
        new_p = (p[0] + movement * correction_direction[0], p[1] + movement * correction_direction[1])

        if "closest_wall_sized" in object_model and object_model["closest_wall_sized"] and available_wall_length is not None:
            # The CAD model may be scaled to fit the size of closest wall
            if (closest_wall_length - available_wall_length) / closest_wall_length < house_gen_conf.def_object_fit_entire_wall_ratio:
                # We are almost fitting the entire wall. So why not fit the entire wall?

                scale_x = (closest_wall_length - house_gen_conf.def_wall_depth) / object_model["dx"]
                scale_x = min(scale_x, 1)
                closest_wall_center = ((closest_wall.p1[0] + closest_wall.p2[0]) / 2, (closest_wall.p1[2] + closest_wall.p2[2]) / 2)
                center_movement = (closest_wall_center[0] - closest_point[0], closest_wall_center[1] - closest_point[1])
                new_p = (new_p[0] + center_movement[0], new_p[1] + center_movement[1])
            else:
                scale_x = available_wall_length / object_model["dx"]
                scale_x = min(scale_x, 1)

        # Compute rotation of object
        object_angle = find_angle((correction_direction[0], -correction_direction[1])) + 90.0
        object_pos_x = new_p[0]
        object_pos_z = new_p[1]
    else:  # No clamp wall
        if "fit_box" in object_model and object_model["fit_box"]:
            # CAD model should fit the AABB bounding box
            box_long = max(abs(obj.p1[0] - obj.p2[0]), abs(obj.p1[1] - obj.p2[1]))
            box_short = min(abs(obj.p1[0] - obj.p2[0]), abs(obj.p1[1] - obj.p2[1]))
            if object_model["ratio"] == "wide":
                scale_x = box_long / object_model["dx"]
                scale_z = box_short / object_model["dz"]
            else:
                scale_x = box_short / object_model["dx"]
                scale_z = box_long / object_model["dz"]

        if "hole_sized" in object_model and object_model["hole_sized"]:
            # CAD model should be scaled to fit the size of hole in wall
            if len(hole_sizes) > 0:
                scale_x = hole_sizes[0] / object_model["dx"]

        wall_mask = get_room_mask(house.rooms[contained_room_id])
        object_angle = compute_object_rotation(obj.p1, obj.p2,
                                               object_model["ratio"],
                                               object_model["face_hole"],
                                               room_walls,
                                               wall_mask)
        object_pos_x = (obj.p1[0] + obj.p2[0]) / 2
        object_pos_z = (obj.p1[1] + obj.p2[1]) / 2

    # Apply Scale bounds
    if "min_scale_x" in object_model:
        scale_x = max(object_model["min_scale_x"], scale_x)

    if "min_scale_z" in object_model:
        scale_z = max(object_model["min_scale_z"], scale_z)

    # Apply uniform scale
    if "uniform_scale" in object_model and object_model["uniform_scale"]:
        scale_x = min(scale_x, scale_z)
        scale_z = scale_x

    # Compute transform matrix
    object_transform = get_transform(object_pos_x, lift, object_pos_z, object_angle,
                                     scale_x=scale_x, scale_z=scale_z)

    object_json = {
        "modelId": object_model["model"],
        "index": index,
        "parentIndex": -1,
        "transform": {
            "rows": 4,
            "cols": 4,
            "data": object_transform,
        }
    }
    index += 1
    return object_json, index


def find_suitable_cad_model(obj: ObjectAnnotation, closest_wall: Wall, house: House, house_gen_conf: Config,
                            room_types: list, room_walls: list, hole_sizes: list, available_wall_length: float, max_box_length: float) -> dict:
    """
    Find suitable CAD model for an object annotation.
    :param obj: Object Annotation
    :param closest_wall: Closest wall to the object
    :param house: House
    :param house_gen_conf: Default configurations
    :param room_types: Room types
    :param room_walls: Walls of the room
    :param hole_sizes: Sizes of holes in the room
    :param available_wall_length: Length available to closest edge of clamped wall
    :param max_box_length: Maximum length of object AABB
    :return: Suitable CAD model or None
    """

    # Extend this method to improve the CAD model selection logic

    if obj.type not in house_gen_conf.fixed_object_model_defaults.__dict__:
        return None

    original_candidates = house_gen_conf.fixed_object_model_defaults[obj.type]
    candidates = [a for a in original_candidates if
                  "allowed_rooms" in a and a["allowed_rooms"] is not None and len(set(room_types).intersection(set(a["allowed_rooms"]))) > 0]
    if len(candidates) == 0:
        # No specific candidates. Try general candidates
        candidates = [a for a in original_candidates if "allowed_rooms" not in a or a["allowed_rooms"] is None]
    if len(candidates) == 0:
        return None

    # Specific rules
    if obj.type in house_gen_conf.object_type_specific_rules.sort_candidates_using_dx_max_aabb_length_difference:
        # Among candidates, select candidate with closest dx to the annotation max length
        candidates = sorted(candidates, key=lambda a: abs(a["dx"] - max_box_length))
        return candidates[0]

    return candidates[0]


def place_object_models(house: House, object_annotations: list, scene_json, house_gen_conf: Config) -> None:
    """
    Add CAD models of objects to the scene.json
    :param house: House considered
    :param object_annotations: Object annotations for the house
    :param scene_json: Scene.json of the house
    :param house_gen_conf: house_gen config used
    """
    room_key_objects_map = assign_objects_to_rooms(house.rooms, object_annotations, arch_defaults=house_gen_conf)

    index = 0
    if len(scene_json["scene"]["object"]) > 0:
        index = max([a["index"] for a in scene_json["scene"]["object"]]) + 1

    new_object_jsons = []
    for obj in object_annotations:
        contained_rooms = [a for a, v in house.rooms.items() if obj in room_key_objects_map[v.room_id]]
        assert len(contained_rooms) == 1, len(contained_rooms)

        object_json, index = get_object_json(obj, house, contained_rooms[0], house_gen_conf, index)
        if object_json is not None:
            new_object_jsons.append(object_json)
    scene_json["scene"]["object"].extend(new_object_jsons)


def parse_object_annotations_json(object_annotations_json_path) -> list:
    """
    Parse an objectaabb.json file
    :param object_annotations_json_path:
    :return:
    """
    with open(object_annotations_json_path) as f:
        objects_json = json.load(f)

    results = []
    for object_json in objects_json["objects"]:
        bound_box = (object_json["bound_box"]["p1"][0], object_json["bound_box"]["p1"][1], object_json["bound_box"]["p2"][0], object_json["bound_box"]["p2"][1])
        object_type = object_json["type"]
        results.append(ObjectAnnotation(bound_box, object_type))

    return results


if __name__ == "__main__":
    """
    Update scene.json file by placing CAD models for object annotations.
    """
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Update scene.json file by placing CAD models for object annotations.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Path to create cad model placed scene.json files.")
    parser.add_argument("house_jsons_path", help="Path to scene.jsons without objects.")
    parser.add_argument("objectabbb_jsons_path", help="Path to objectavv,json files.")
    parser.add_argument("--remove-existing-objects", default=False, action="store_true")
    parser.add_argument("--texture-internal-walls-only", action="store_true", default=False, help="Specify flag to ommit textures on external side of perimeter walls.")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    houses_path = args.house_jsons_path
    object_annotations_path = args.objectabbb_jsons_path
    remove_existing_objects = args.remove_existing_objects
    texture_internal_walls_only = args.texture_internal_walls_only

    house_files = os.listdir(houses_path)
    house_files = [a for a in house_files if a.endswith(".scene.json")]

    for i, house_file in enumerate(house_files):
        logging.info("[{i}/{count}]Processing {house_file}".format(i=i, count=len(house_files), house_file=house_file))

        house = parse_house_json_file(osp.join(houses_path, house_file), None)
        object_annotations = parse_object_annotations_json(osp.join(object_annotations_path, house.house_key + ".objectaabb.json"))

        if remove_existing_objects:
            house.cad_models.clear()

        scene_json = serialize_scene_json(house, texture_both_sides_of_walls=not texture_internal_walls_only)
        place_object_models(house, object_annotations, scene_json, house_gen_conf=conf.house_gen)

        # Save
        save_file_name = None
        if house_file.endswith(".arch.json"):
            save_file_name = osp.splitext(osp.splitext(house_file)[0])[0] + ".scene.json"
        elif house_file.endswith(".scene.json"):
            save_file_name = house_file
        assert not osp.exists(osp.join(output_path, save_file_name)), "Output file already exists: {f}. Please delete it.".format(
            f=osp.join(output_path, save_file_name))
        with open(osp.join(output_path, save_file_name), "w") as f:
            json.dump(scene_json, f, indent=4)
        logging.info("Saved {file}".format(file=save_file_name))
