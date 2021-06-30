from arch_parser import json_util
from arch_parser.models.cad_model import CADModel
from arch_parser.models.house import House
from PIL import Image
import copy

from arch_parser.models.room import Room
from arch_parser.models.wall_room_assignment import WallRoomAssignment

NA_COLOR = "#f6deff"


def serialize_texture(texture: Image, surface_id: str, materials_json: list, textures_json: list,
                      images_json: list) -> dict:
    """
    Serializes a given image as a texture. Updates materials_json, textures_json and images_json to register the texture.
    Returns the texture material definition that should used to assign the texture.

    :param texture: Texture to be serialized
    :param surface_id: Surface id used to recognize texture
    :param materials_json: Materials json which will get updated
    :param textures_json: Textures json which will get updated
    :param images_json: Images json which will get updated
    :return: Material definition
    """
    if texture is None:
        return {
            "name": "surface",
            "diffuse": NA_COLOR
        }

    mat_json = {"uuid": f"Material_{surface_id}", "map": f"Texture_{surface_id}"}
    materials_json.append(mat_json)

    tex_json = {"uuid": f"Texture_{surface_id}", "image": f"Image_{surface_id}"}
    textures_json.append(tex_json)

    image_data_url = json_util.pil_to_data_url(texture)
    image_json = {"uuid": f"Image_{surface_id}", "url": image_data_url}
    images_json.append(image_json)

    return {
        "name": "surface",
        "materialId": f"Material_{surface_id}"
    }


def serialize_room(room: Room, elements: list, materials_json: list, textures_json: list, images_json: list, added_walls: dict,
                   texture_wall_both_sides: bool) -> dict:
    """
    Serializes a room, updating the elements list, materials_json, textures_json and images_json
    :param room: Room to be serialized
    :param elements: List of elements to be updated
    :param materials_json: List of materials to be updated
    :param textures_json: List of textures to be updated
    :param images_json: List of images to be updated
    :param added_walls: Dictionary of already added walls
    :param texture_wall_both_sides: Specify true to copy the texture to both inside and outside surface of a wall
    :return: Serialized room json
    """

    floor_json = json_util.generate_floor_json(room.floor, room.room_id,
                                               serialize_texture(room.floor_texture, room.floor.id, materials_json,
                                                                 textures_json, images_json))
    elements.append(floor_json)

    ceiling_json = json_util.generate_ceiling_json(room.ceiling, room.room_id,
                                                   serialize_texture(room.ceiling_texture, room.ceiling.id,
                                                                     materials_json,
                                                                     textures_json, images_json))
    elements.append(ceiling_json)

    for wall_assignment in room.walls:
        assert isinstance(wall_assignment, WallRoomAssignment)
        wall = wall_assignment.wall
        inner_surface_index = wall_assignment.inner_wall_index

        if wall in added_walls:
            # The wall is already added by another room. We update that wall by assigning the material to the appropriate side.
            assert added_walls[wall]["roomId"][inner_surface_index] is None
            added_walls[wall]["roomId"][inner_surface_index] = room.room_id
            surface_material = copy.deepcopy(serialize_texture(room.wall_texture, wall.wall_id, materials_json,
                                                               textures_json, images_json))
            surface_material["name"] = "surface%d" % inner_surface_index
            added_walls[wall]["materials"][inner_surface_index] = surface_material
        else:
            # New wall
            wall_json = json_util.generate_wall_json(wall, room.room_id, inner_surface_index,
                                                     material=serialize_texture(room.wall_texture, wall.wall_id, materials_json,
                                                                                textures_json, images_json),
                                                     outer_material=serialize_texture(None, wall.wall_id, materials_json,
                                                                                      textures_json, images_json),
                                                     texture_both_sides=texture_wall_both_sides
                                                     )
            added_walls[wall] = wall_json
            elements.append(wall_json)

    room_json = {
        "id": room.room_id,
        "types": room.types
    }
    return room_json


def serialize_rdr(house: House) -> list:
    """
    Serializes RDR to json
    :param house: House containing the rdr
    :return: rdr json
    """
    results = []
    for from_room_id, hole_id, to_room_id in house.door_connected_room_pairs:
        results.append((from_room_id, hole_id, to_room_id))
    return results


def serialize_arch_json(house: House, texture_both_sides_of_walls) -> dict:
    """
    Serializes a house into a arch_json
    :param house: House to be serialized
    :param texture_both_sides_of_walls: Both sides of all walls are textured, including walls with only one interior side. The interior side texture is copied to exterior side.
    :return: arch_json as a dict
    """
    elements = []

    # Textures
    materials_json = []
    textures_json = []
    images_json = []

    rooms = []

    added_walls = {}
    for room_id, room in house.rooms.items():
        rooms.append(serialize_room(room, elements, materials_json, textures_json, images_json, added_walls, texture_both_sides_of_walls))

    r = {
        "id": house.house_key,
        "elements": elements,
        "rdr": serialize_rdr(house),
        "rooms": rooms,
        "materials": materials_json,
        "textures": textures_json,
        "images": images_json
    }

    # Add extra fields
    for extra_meta in house.arch_extra_metadata:
        r[extra_meta] = house.arch_extra_metadata[extra_meta]

    return r


def serialize_cad_models(house: House) -> list:
    """
    Serializes CAD models in a house.
    :param house: House with CAD models.
    :return: object list of the scene.json file.
    """
    object_jsons = []
    for cad_model in house.cad_models:
        assert isinstance(cad_model, CADModel)
        object_jsons.append({
            "modelId": cad_model.model_id,
            "index": cad_model.index,
            "parentIndex": cad_model.parent_index,
            "transform": cad_model.transform
        })
    return object_jsons


def serialize_scene_json(house: House, texture_both_sides_of_walls) -> dict:
    """
    Serializes a house into a arch_json
    :param house: House to be serialized
    :param texture_both_sides_of_walls: Both sides of all walls are textured, including walls with only one interior side. The interior side texture is copied to exterior side.
    :return: arch_json as a dict
    """
    arch_json = serialize_arch_json(house, texture_both_sides_of_walls)
    object_jsons = serialize_cad_models(house)
    result = {
        "format": "sceneState",
        "scene": {
            "arch": arch_json,

            # Load defaults from arch
            "object": object_jsons,
            "up": {
                "x": arch_json["up"][0] if "up" in arch_json else 0.0,
                "y": arch_json["up"][1] if "up" in arch_json else 0.0,
                "z": arch_json["up"][2] if "up" in arch_json else 0.0,
            },
            "front": {
                "x": arch_json["front"][0] if "front" in arch_json else 0.0,
                "y": arch_json["front"][1] if "front" in arch_json else 0.0,
                "z": arch_json["front"][2] if "front" in arch_json else 0.0,
            },
            "unit": arch_json["scaleToMeters"] if "scaleToMeters" in arch_json else 0.0
        },
        "selected": [],
    }
    for extra_meta in house.scene_extra_metadata:
        result["scene"][extra_meta] = house.scene_extra_metadata[extra_meta]

    return result
