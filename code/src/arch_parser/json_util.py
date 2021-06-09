import copy
import io

import base64

from PIL import Image

from arch_parser.models.ceiling import Ceiling
from arch_parser.models.floor import Floor
from arch_parser.models.hole import Hole
from arch_parser.models.wall import Wall


def data_url_to_pil(data_url: str) -> Image.Image:
    """
    Converts a data url to a PIL Image
    :param data_url: Data URL to parse.
    :return: Parsed image
    """
    START_TOKEN = u'data:img/png;base64,'
    assert data_url.startswith(START_TOKEN)
    data64 = data_url[len(START_TOKEN):]
    data64 = data64.encode('utf-8')
    data64 = base64.b64decode(data64)
    data = io.BytesIO(data64)
    img = Image.open(data)
    return img


def pil_to_data_url(img: Image.Image) -> str:
    """
    Converts PIL image to data url
    :param img: PIL Image to be converted
    :return: Data URL
    """
    # From https://gist.github.com/ankitshekhawat/56f05d0fb0453a0cd143630ab5af5087

    # Converts PIL image to DataURL

    data = io.BytesIO()
    img.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/png;base64,' + data64.decode('utf-8')


def generate_hole_json(hole: Hole) -> dict:
    """
    Generates json for a hole
    :param hole: Hole to be processed
    :return: hole json
    """
    hole_json = {
        "id": hole.hole_id,  # Node id of object creating hole in the wall
        "type": hole.hole_type,  # hole type (`Door` or `Window`)
        "box": {  # cutout of hole as box on the wall
            "min": [hole.start, hole.min_height],  # minimum point
            # x is distance from points[0] (toward points[1])
            # y is height from wall bottom (goes from 0 to wall height)
            "max": [hole.end, hole.max_height]  # maximum point
        }
    }
    for extra_arg in hole.extra_args:
        hole_json[extra_arg] = hole.extra_args[extra_arg]

    return hole_json


def generate_wall_json(wall: Wall, room_id: str, material: dict) -> dict:
    """
    Generate json for a wall of a house
    :param wall:
    :param room_id:
    :param material:
    :return:
    """

    hole_jsons = []

    for hole_index, hole in enumerate(wall.holes):
        hole_jsons.append(generate_hole_json(hole))

    surface1_material = copy.deepcopy(material)
    surface1_material["name"] = "surface1"

    surface2_material = copy.deepcopy(material)
    surface2_material["name"] = "surface2"

    wall_json = {
        "roomId": room_id,
        "id": wall.wall_id,
        "type": "Wall",
        "points": [wall.p1, wall.p2],
        "holes": hole_jsons,
        "materials": [surface1_material, surface2_material],  # inner and outer material
    }

    for extra_arg in wall.extra_args:
        wall_json[extra_arg] = wall.extra_args[extra_arg]

    return wall_json


def generate_ceiling_json(ceiling: Ceiling, room_id: str, material: dict) -> dict:
    """
    Generate json for a ceiling of a house
    :param ceiling: Ceiling to be parsed
    :param room_id: room_id of associated room
    :param material: Material dict
    :return: Ceiling json
    """

    r = {
        "id": ceiling.id,
        "roomId": room_id,
        "points": [[p for p in ceiling.points]],
        "type": "Ceiling",
        "materials": [material]
    }
    for extra_arg in ceiling.extra_args:
        r[extra_arg] = ceiling.extra_args[extra_arg]
    return r


def generate_floor_json(floor: Floor, room_id: str, material: dict) -> dict:
    """
    Generate json for a floor of a house
    :param floor: Floor to be parsed
    :param room_id: room_id of associated room
    :param material: Material dict
    :return: Floor json
    """

    r = {
        "id": floor.id,
        "roomId": room_id,
        "points": [[p for p in floor.points]],
        "type": "Floor",
        "materials": [material]
    }
    for extra_arg in floor.extra_args:
        r[extra_arg] = floor.extra_args[extra_arg]
    return r
