import logging

from arch_parser.models.cad_model import CADModel
from arch_parser.models.hole import Hole
from arch_parser.models.house import House
from arch_parser.models.room import Room
from arch_parser.models.wall_room_assignment import WallRoomAssignment
from arch_parser.parser import parse_house_json_file
from arch_parser.serializer import serialize_scene_json
from plan2scene.house_gen.geom_util import hole_to_line, get_transform
from plan2scene.config_manager import ConfigManager
import os.path as osp
import os
import json
import math


def get_hole_model(conf: ConfigManager, hole_type, length, adjacent_rooms):
    """
    Identify the most suitable CAD model to fill a hole (door/window).
    :param conf: ConfigManager
    :param hole_type: Window or Door
    :param length: Length of the hole
    :param adjacent_rooms: Adjacent rooms
    :return: Description of a hole model as specified by conf.house_gen.hole_model_defaults
    """
    # TODO: Improve logic here
    if hole_type in conf.house_gen.hole_model_defaults.__dict__:
        entrance = False

        if len(adjacent_rooms) <= 1:
            entrance = True

        balcony = False
        if len([a for a in adjacent_rooms if len([b for b in a.types if b in conf.house_gen.outdoor_room_types]) > 0]) > 0:
            balcony = True

        bathroom = False
        if len([a for a in adjacent_rooms if len([b for b in a.types if b in conf.house_gen.bathroom_room_types]) > 0]) > 0:
            bathroom = True

        candidates = conf.house_gen.hole_model_defaults.__dict__[hole_type]
        candidates = [a for a in candidates if entrance in a["entrance"]]
        candidates = [a for a in candidates if balcony in a["balcony"]]
        candidates = [a for a in candidates if bathroom in a["bathroom"]]
        length_sorted = sorted(candidates, key=lambda a: abs(a["length"] - length))
        if len(length_sorted) > 0:
            return length_sorted[0]
    return None


def generate_hole_cad_placements(conf: ConfigManager, house: House, start_index: int):
    """
    Obtain list of CADModels that should be placed for holes of the house.
    :param conf: Config Manager
    :param house: House populated
    :param start_index: CAD models will be indexed starting from start_index
    :return: List of CADModel
    """
    cad_models = []
    index = start_index
    for room_key, room in house.rooms.items():
        assert isinstance(room, Room)
        for i_wall, wall_assignment in enumerate(room.walls):
            assert isinstance(wall_assignment, WallRoomAssignment)
            wall = wall_assignment.wall
            new_wall_width = math.sqrt((wall.p2[1] - wall.p1[1]) ** 2 + (wall.p2[0] - wall.p1[0]) ** 2)

            for hole_index, hole in enumerate(wall.holes):
                assert isinstance(hole, Hole)
                hole_min_x = min(hole.start, hole.end)
                hole_max_x = max(hole.start, hole.end)
                hole_length = hole_max_x - hole_min_x
                assert hole_length > 0

                adjacent_rooms = [room]
                # Find other adjacent room using door_connected_room_pairs map
                adjacent_candidates = [house.rooms[a[2]] for a in house.door_connected_room_pairs if
                                       a[1] == hole.hole_id and a[0] == room.room_id and a[2] is not None]
                adjacent_rooms.extend(adjacent_candidates)

                adjacent_candidates = [house.rooms[a[0]] for a in house.door_connected_room_pairs if
                                       a[1] == hole.hole_id and a[2] == room.room_id and a[0] is not None]
                adjacent_rooms.extend(adjacent_candidates)

                # Identify suitable CAD model
                hole_model = get_hole_model(conf, hole.hole_type, hole_length, adjacent_rooms)

                # Identify placement of CAD model
                if hole_model is not None:
                    model_id = hole_model["model"]

                    # Adjust hole to fit the CAD model
                    hole.min_height = float(hole_model["hole_min_y"])
                    hole.max_height = float(hole_model["hole_max_y"])

                    model_length = hole_model["length"]
                    model_lift = hole_model["lift"]

                    # Identify position
                    wall_p1 = (wall.p1[0], wall.p1[2])
                    wall_p2 = (wall.p2[0], wall.p2[2])
                    (start_x, start_z), (end_x, end_z) = hole_to_line(wall_p1, wall_p2, hole.start, hole.end)
                    door_x = (start_x + end_x) / 2
                    door_z = (start_z + end_z) / 2

                    # Identify angle
                    object_angle = 90.0
                    if start_x != end_x:
                        object_angle = math.atan((end_z - start_z) / (end_x - start_x)) / math.pi * 180.0

                    if "rotate180" in hole_model and hole_model["rotate180"]:
                        object_angle += 180.0
                    object_angle = object_angle % 360

                    # Identify scale
                    scale_x = float(hole_length) / model_length

                    # Generate CAD model placement
                    cad_model = CADModel(model_id=model_id, index=index, parent_index=-1, transform={
                        "rows": 4,
                        "cols": 4,
                        "data": get_transform(door_x, model_lift, door_z, object_angle,
                                              scale_x=scale_x),
                    })

                    index += 1
                    cad_models.append(cad_model)

    return cad_models, index


def process_house(conf: ConfigManager, house: House) -> None:
    """
    Place hole CAD models for the given house.
    :param conf: Config Manager
    :param house: House processed.
    """
    index = 0
    if len(house.cad_models) > 0:
        index = max([a.index for a in house.cad_models]) + 1
    cad_models, index = generate_hole_cad_placements(conf, house, index)
    house.cad_models.extend(cad_models)


if __name__ == "__main__":
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Place CAD models for holes (windows/doors) of walls.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Path to save cad model placed scene.json files.")
    parser.add_argument("house_jsons_path", help="Path to scene.json files without objects.")
    parser.add_argument("--remove-existing-objects", default=False, action="store_true", help="Specify true to clear any existing CAD models.")
    parser.add_argument("--texture-internal-walls-only", action="store_true", default=False,
                        help="Specify flag to ommit textures on external side of perimeter walls.")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    houses_path = args.house_jsons_path
    remove_existing_objects = args.remove_existing_objects
    texture_internal_walls_only = args.texture_internal_walls_only

    house_files = os.listdir(houses_path)
    house_files = [a for a in house_files if a.endswith(".scene.json")]

    for i, house_file in enumerate(house_files):
        logging.info("[{i}/{count}]Processing {house_file}".format(i=i, count=len(house_files), house_file=house_file))

        house = parse_house_json_file(osp.join(houses_path, house_file), None)
        if remove_existing_objects:
            house.cad_models.clear()

        process_house(conf, house)
        scene_json = serialize_scene_json(house, texture_both_sides_of_walls=not texture_internal_walls_only)

        save_file_name = None
        if house_file.endswith(".arch.json"):
            save_file_name = osp.splitext(osp.splitext(house_file)[0])[0] + ".scene.json"
        elif house_file.endswith(".scene.json"):
            save_file_name = house_file
        with open(osp.join(output_path, save_file_name), "w") as f:
            json.dump(scene_json, f, indent=4)
