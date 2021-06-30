import json
import pandas as pd

from arch_parser.json_util import data_url_to_pil
from arch_parser.models.cad_model import CADModel
from arch_parser.models.ceiling import Ceiling
from arch_parser.models.floor import Floor
from arch_parser.models.hole import Hole
from arch_parser.models.house import House
from arch_parser.models.room import Room
from arch_parser.models.wall import Wall
from arch_parser.models.wall_room_assignment import WallRoomAssignment
from arch_parser.preferred_format import PreferredFormat
from PIL import Image


def parse_photo_assignments(photoroom_df: pd.DataFrame, house: House) -> None:
    """
    Parses photo to room assignments.
    :param photoroom_df: Dataframe defining photo room assignments.
    :param house: House that gets updated.
    """
    for i in range(len(photoroom_df)):
        entry = photoroom_df.iloc[i]
        house.rooms[entry.roomId].photos.append(entry.photo)


def parse_rdr(rdr_json: list, house: House) -> None:
    """
    Parses room-door-room connectivity graph.
    :param rdr_json: List of room door room edges.
    :param house: House that gets updated.
    """
    for from_room_id, hole_id, to_room_id in rdr_json:
        house.door_connected_room_pairs.append((from_room_id, hole_id, to_room_id))


def parse_rooms(rooms_json: list, house: House) -> None:
    """
    Parses room definitions of an arch.json.
    :param rooms_json: List of room definition
    :param house: House that gets updated.
    """
    for room_json in rooms_json:
        room = Room(house.house_key, room_json["id"])
        for room_type in room_json["types"]:
            room.types.append(room_type)
        house.rooms[room.room_id] = room


def parse_material(material_entry: dict, arch_json: dict) -> Image.Image:
    """
    Parses a material assigned to an arch surface.
    :param material_entry: Entry { name: '', materialId: '' } assigning a material to a surface.
    :param arch_json: Arch json of the house.
    :return: Parsed material as an Image
    """
    if "materialId" not in material_entry:
        return None

    # Parse material entry
    mat_entry = [a for a in arch_json["materials"] if a["uuid"] == material_entry["materialId"]]
    assert len(mat_entry) == 1
    mat_entry = mat_entry[0]

    # Parse texture entry
    texture_entry = [a for a in arch_json["textures"] if a["uuid"] == mat_entry["map"]]
    assert len(texture_entry) == 1
    texture_entry = texture_entry[0]

    # Parse image entry
    image_entry = [a for a in arch_json["images"] if a["uuid"] == texture_entry["image"]]
    assert len(image_entry) == 1
    image_entry = image_entry[0]

    image = data_url_to_pil(image_entry["url"])
    return image


def parse_floor_element(element_json: dict, arch_json: dict, house: House) -> None:
    """
    Parses floor element.
    :param element_json: Floor element json from arch.json.
    :param arch_json: Arch json.
    :param house: House that gets updated.
    """
    room_id = element_json["roomId"]
    floor_points = [tuple(p) for p in element_json["points"][0]]
    extra_args = {k: v for k, v in element_json.items() if k not in ["points", "roomId", "id", "type", "materials"]}
    floor = Floor(floor_id=element_json["id"], points=floor_points, extra_args=extra_args)

    room = house.rooms[room_id]
    assert isinstance(room, Room)

    if len(element_json["materials"]) > 0:
        room.floor_texture = parse_material(element_json["materials"][0], arch_json)

    assert room.floor is None
    house.rooms[room_id].floor = floor


def parse_hole(hole_json: dict, wall: Wall) -> None:
    """
    Parses a hole in a wall.
    :param hole_json: Hole json.
    :param wall: Wall that gets updated.
    """
    hole_id = hole_json["id"]
    hole_type = hole_json["type"]
    min_height = hole_json["box"]["min"][1]
    max_height = hole_json["box"]["max"][1]
    start = hole_json["box"]["min"][0]
    end = hole_json["box"]["max"][0]
    extra_args = {k: v for k, v in hole_json.items() if k not in ["box", "id", "type"]}

    hole = Hole(hole_id=hole_id, hole_type=hole_type, start=start, end=end, min_height=min_height, max_height=max_height,
                extra_args=extra_args)
    wall.holes.append(hole)


def parse_wall_element(element_json: dict, arch_json: dict, house: House) -> None:
    """
    Parses a wall element.
    :param element_json: Wall element in arch.json file.
    :param arch_json: Arch json
    :param house: House that gets updated.
    """
    wall_points = [tuple(p) for p in element_json["points"]]
    assert len(wall_points) == 2

    extra_args = {k: v for k, v in element_json.items() if k not in ["points", "holes", "roomId", "id", "type", "materials"]}
    wall = Wall(wall_id=element_json["id"], p1=tuple(wall_points[0]), p2=tuple(wall_points[1]), extra_args=extra_args)

    for hole_json in element_json["holes"]:
        parse_hole(hole_json, wall)

    room_ids = element_json["roomId"]
    if not isinstance(room_ids, list):
        # Convert to two rooms per wall format
        room_ids = [None, room_ids]
    
    for i, room_id in enumerate(room_ids):
        if room_id is None:
            continue
        
        room = house.rooms[room_id]
        assert isinstance(room, Room)
        if len(element_json["materials"]) > 0:
            room.wall_texture = parse_material(element_json["materials"][i], arch_json)

        house.rooms[room_id].walls.append(WallRoomAssignment(wall, i))


def parse_ceiling_element(element_json: dict, arch_json: dict, house: House) -> None:
    """
    Parses a ceiling element.
    :param element_json: Ceiling element from arch.json.
    :param arch_json: Arch json.
    :param house: House that gets updated.
    """
    room_id = element_json["roomId"]
    ceiling_points = [tuple(p) for p in element_json["points"][0]]
    extra_args = {k: v for k, v in element_json.items() if k not in ["points", "roomId", "id", "type", "materials"]}
    ceiling = Ceiling(ceiling_id=element_json["id"], points=ceiling_points, extra_args=extra_args)

    room = house.rooms[room_id]
    assert isinstance(room, Room)

    if len(element_json["materials"]) > 0:
        room.ceiling_texture = parse_material(element_json["materials"][0], arch_json)

    assert house.rooms[room_id].ceiling is None
    house.rooms[room_id].ceiling = ceiling


def parse_elements(elements_json: list, arch_json: dict, house: House) -> None:
    """
    Parses elements list of an arch.json file. Updates house.
    :param elements_json: List of elements to parse.
    :param arch_json: Arch json of the house.
    :param house: House that gets updated.
    """
    for element_json in elements_json:
        if element_json["type"] == "Floor":
            parse_floor_element(element_json, arch_json, house)
        elif element_json["type"] == "Ceiling":
            parse_ceiling_element(element_json, arch_json, house)
        elif element_json["type"] == "Wall":
            parse_wall_element(element_json, arch_json, house)


def parse_arch_json_from_file(arch_json_path: str, photoroom_csv_path: str) -> House:
    """
    Parses a house given the arch.json and the photoroom.csv
    :param arch_json_path: str: Path to arch.json
    :param photoroom_csv_path: str: Path to photoroom.csv
    :return: Parsed house
    """

    with open(arch_json_path) as f:
        arch_json = json.load(f)

    photoroom_df = None
    if photoroom_csv_path is not None:
        with open(photoroom_csv_path) as f:
            photoroom_df = pd.read_csv(f)

    house = parse_arch_json(arch_json, photoroom_df)
    house.preferred_format = PreferredFormat.ARCH_JSON
    return house


def parse_arch_json(arch_json: dict, photoroom_df: pd.DataFrame) -> House:
    """
    Parses an arch_json to a house.
    :param arch_json: Arch json as a dict.
    :param photoroom_df: CSV describing photo room assignments.
    :return: Parsed house.
    """
    house_key = arch_json["id"]
    result = House(house_key)
    parse_rooms(arch_json["rooms"], result)
    parse_elements(arch_json["elements"], arch_json, result)
    parse_rdr(arch_json["rdr"], result)
    if photoroom_df is not None:
        parse_photo_assignments(photoroom_df, result)

    for k, v in arch_json.items():
        if k in ["elements", "rdr", "rooms", "id"]:
            pass  # Handled above
        elif k in ["materials", "textures", "images"]:
            pass  # Already processed when the assigned element got processed
        else:
            # Just pass through
            result.arch_extra_metadata[k] = v

    return result


def parse_object_jsons(object_jsons: list) -> list:
    """
    Parsed 'object' entries of a scene.json.
    :param object_jsons: 'object' entries of a scene.json file.
    :return: Parsed list of objects.
    """
    cad_models = []
    for object_json in object_jsons:
        cad_model = CADModel(model_id=object_json["modelId"], index=object_json["index"], parent_index=object_json["parentIndex"],
                             transform=object_json["transform"])
        cad_models.append(cad_model)
    return cad_models


def parse_scene_json(scene_json: dict, photoroom_df: pd.DataFrame) -> House:
    """
    Parses a scene.json file from memory.
    :param scene_json: Scene json as a dictionary
    :param photoroom_df: Photo room data-frame.
    :return: Parsed house.
    """
    assert scene_json["format"] == "sceneState"

    arch_json = scene_json["scene"]["arch"]
    object_jsons = scene_json["scene"]["object"]

    # Process arch.json
    result = parse_arch_json(arch_json, photoroom_df)
    result.cad_models = parse_object_jsons(object_jsons)

    result.preferred_format = PreferredFormat.SCENE_JSON

    # Backup extras
    for k, v in scene_json["scene"].items():
        if k in ["arch", "object"]:
            pass  # Handled separately above
        else:
            # Just pass through
            result.scene_extra_metadata[k] = v
    return result


def parse_scene_json_from_file(scene_json_path: str, photoroom_csv_path: str) -> House:
    """
    Parses a scene.json file from disk.
    :param scene_json_path: Path to scene.json file.
    :param photoroom_csv_path: Path photoroom.csv file.
    :return: Parsed house.
    """
    with open(scene_json_path) as f:
        scene_json = json.load(f)
    photoroom_df = None
    if photoroom_csv_path is not None:
        with open(photoroom_csv_path) as f:
            photoroom_df = pd.read_csv(f)

    return parse_scene_json(scene_json, photoroom_df)


def parse_house_json_file(house_json_path: str, photoroom_csv_path: str) -> House:
    """
    Parses a scene.json file or an arch.json file. Identifies file type based on extension.
    :param house_json_path: Path to arch.json / scene.json file.
    :param photoroom_csv_path: Path to photoroom.csv file.
    :return: Parsed house.
    """
    if house_json_path.endswith(".arch.json"):
        return parse_arch_json_from_file(house_json_path, photoroom_csv_path)
    elif house_json_path.endswith(".scene.json"):
        return parse_scene_json_from_file(house_json_path, photoroom_csv_path)
    assert False