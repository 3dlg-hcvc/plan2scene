from arch_parser import json_util
from arch_parser.models.cad_model import CADModel
from arch_parser.models.house import House
from PIL import Image

from arch_parser.models.room import Room

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


def serialize_room(room: Room, elements: list, materials_json: list, textures_json: list, images_json: list) -> dict:
    """
    Serializes a room, updating the elements list, materials_json, textures_json and images_json
    :param room: Room to be serialized
    :param elements: List of elements to be updated
    :param materials_json: List of materials to be updated
    :param textures_json: List of textures to be updated
    :param images_json: List of images to be updated
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

    for wall in room.walls:
        wall_json = json_util.generate_wall_json(wall, room.room_id,
                                                 serialize_texture(room.wall_texture, wall.wall_id, materials_json,
                                                                   textures_json, images_json))
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


def serialize_arch_json(house: House) -> dict:
    """
    Serializes a house into a arch_json
    :param house: House to be serialized
    :return: arch_json as a dict
    """
    elements = []

    # Textures
    materials_json = []
    textures_json = []
    images_json = []

    rooms = []

    for room_id, room in house.rooms.items():
        rooms.append(serialize_room(room, elements, materials_json, textures_json, images_json))

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


def serialize_scene_json(house: House) -> dict:
    """
    Serializes a house into a arch_json
    :param house: House to be serialized
    :return: arch_json as a dict
    """
    arch_json = serialize_arch_json(house)
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


if __name__ == "__main__":
    # Test

    from arch_parser.parser import parse
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Parse and serialize a house")
    parser.add_argument("output_path", help="Path to save output arch.json")
    parser.add_argument("arch_json_path", help="arch.json path")
    parser.add_argument("photoroom_csv_path", help="photoroom.csv path")
    args = parser.parse_args()

    house = parse(args.arch_json_path, args.photoroom_csv_path)

    for room_id, room in house.rooms.items():
        room.floor_texture = Image.new("RGB", (128, 128), color=(255, 0, 0))
        room.wall_texture = Image.new("RGB", (128, 128), color=(0, 255, 0))
        room.ceiling_texture = Image.new("RGB", (128, 128), color=(0, 0, 255))

    serialized_json = serialize_arch_json(house)
    import json

    r = json.dumps(serialized_json, indent=4)
    print(r)
    with open(args.output_path, "w") as f:
        f.write(r)
