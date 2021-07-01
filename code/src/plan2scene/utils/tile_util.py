import shutil
import subprocess
import os
import os.path as osp
import uuid
from PIL import Image
import tempfile
import subprocess

from plan2scene.common.image_description import ImageDescription
from plan2scene.common.residence import House, Room
from plan2scene.utils.io import load_image


def tile_image(image: Image.Image, embark_texture_synthesis_path: str, seam_mask_path: str):
    """
    Correct seams of a texture so it can be tiled
    :param image: Texture to be seam corrected
    :param embark_texture_synthesis_path: Path to texture-synthesis project.
    :param seam_mask_path: Path to texture-synthesis/imgs/masks/1_tile.jpg
    :return: Seam corrected image
    """
    assert isinstance(image, Image.Image)
    temp_location = tempfile.mkdtemp()
    try:
        prefix = uuid.uuid4().hex
        script_path = osp.abspath(embark_texture_synthesis_path)
        script_dir_path = osp.dirname(script_path)
        image.save(osp.join(temp_location, prefix + "_to_tile.png"))
        command = "%s --inpaint %s --out-size %d --tiling -o %s generate %s" % (
            script_path, osp.abspath(seam_mask_path),
            image.width,
            osp.join(temp_location, prefix + "_tiled.png"),
            osp.join(temp_location, prefix + "_to_tile.png"),
        )

        assert subprocess.call(command, shell=True) == 0
        tiled_image = load_image(osp.join(temp_location, prefix + "_tiled.png"))
        return tiled_image
    finally:
        shutil.rmtree(temp_location)


def seam_correct_surface(textures_map: dict, embark_texture_synthesis_path: str, seam_mask_path: str, key: str = "prop") -> None:
    """
    Correct seams of textures of a surface.
    :param textures_map: Surface textures dictionary of a surface
    :param embark_texture_synthesis_path:  Path to embark studios texture synthesis library.
    :param seam_mask_path: Path to the mask used for seam correction.
    :param key: Key denoting predicted texture. We seam correct this entry.
    """
    if key in textures_map:
        texture_description = textures_map[key]
        assert isinstance(texture_description, ImageDescription)
        texture = texture_description.image
        texture = tile_image(texture, embark_texture_synthesis_path, seam_mask_path)
        texture_description.image = texture


def seam_correct_house(house: House, embark_texture_synthesis_path: str, seam_mask_path: str) -> None:
    """
    Correct seams of predicted textures assigned to a house.
    :param house: House considered
    :param embark_texture_synthesis_path: Path to embark studios texture synthesis library.
    :param seam_mask_path: Path to the mask used for seam correction.
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        for surface in room.surface_textures:
            seam_correct_surface(room.surface_textures[surface], embark_texture_synthesis_path, seam_mask_path)
