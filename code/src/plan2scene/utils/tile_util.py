import shutil
import subprocess
import os
import os.path as osp
import uuid
from PIL import Image
import tempfile
import subprocess

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

    try:
        temp_location = tempfile.mkdtemp()
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
