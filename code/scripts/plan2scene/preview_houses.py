#!/bin/python3
import logging

from plan2scene.common.house_parser import parse_houses, load_house_crops
import argparse

from plan2scene.common.image_description import ImageSource, ImageDescription
from plan2scene.common.residence import House, Room
from plan2scene.config_manager import ConfigManager
import os.path as osp
import os

# Constants
MARGIN = 10
PREFERRED_SIZE = 400
WALL_COLOR = (255, 0, 0)
FOCUSED_WALL_COLOR = (0, 255, 0)
RENDER_SIZE = 400

HTML_HEADER = """
<head>
<style>
table, td, th {
  border: 1px solid black;
}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
}
</style>
</head>
"""


def preview_house(house: House, gt_house: House, house_json_path: str, photos_path: str, output_path: str) -> None:
    """
    Create a HTML page that previews predictions for a house.
    :param house: House with synthesized texture. Unobserved photos are not present.
    :param gt_house: House having all the photos.
    :param house_json_path: Path to scene.json file.
    :param photos_path: Path to raw photos.
    :param output_path: Path to save the preview page.
    """
    if not osp.exists(output_path):
        os.mkdir(output_path)
    if not osp.exists(osp.join(output_path, "rooms")):
        os.mkdir(osp.join(output_path, "rooms"))

    if not osp.exists(osp.join(output_path, "photos")):
        os.mkdir(osp.join(output_path, "photos"))

    with open(osp.join(output_path, "report.html"), "w") as f:
        f.write("<html>\n")
        f.write(HTML_HEADER)
        f.write("<body>\n")
        f.write("<h1 style='text-align:center;' >House: {house_id}</h1>".format(house_id=house.house_key))
        # Preview house
        if osp.exists(osp.splitext(house_json_path)[0] + ".png"):
            if not osp.exists(osp.join(output_path, "render.png")):
                os.symlink(osp.abspath(osp.splitext(house_json_path)[0] + ".png"), osp.join(output_path, "render.png"))
            f.write("<img class='center' style='width:{render_size}px' src='{render_path}'/>\n".format(render_path="render.png", render_size=RENDER_SIZE))
        else:
            f.write("<p style='text-align:center;'>3D rendering not found.</p>\n")
        f.write("<table>\n")
        # Table header
        columns = ["Room Id", "Room Types", "Sketch<br>Room indicated in green."]
        columns.extend(conf.surfaces)
        columns.append("Photos<br>Red outline: Photo unobserved<br>Green outline: Photo observed")
        f.write("<tr>%s</tr>\n" % "".join(["<th>%s</th>" % s for s in columns]))

        # Row per room
        for room_id, room in house.rooms.items():
            gt_room = gt_house.rooms[room_id]
            assert isinstance(room, Room)
            assert isinstance(gt_room, Room)
            assert gt_room.room_id == room.room_id

            if not osp.exists(osp.join(output_path, "rooms", room.room_id)):
                os.mkdir(osp.join(output_path, "rooms", room.room_id))

            f.write("<tr>\n")
            f.write("<td>%s</td>\n" % str(room.room_id))
            f.write("<td>%s</td>\n" % str(room.types))

            # Sketch house with room highlighted
            house.sketch_house(focused_room_id=room.room_id).save(osp.join(output_path, "rooms", room.room_id, "sketch.png"))
            f.write("<td><img src='{sketch_path}'/></td>\n".format(sketch_path=osp.join("rooms", room.room_id, "sketch.png")))

            # Preview textures
            for surface in conf.surfaces:
                f.write("<td>\n")
                if "prop" in room.surface_textures[surface]:
                    texture_img = room.surface_textures[surface]["prop"]
                    assert isinstance(texture_img, ImageDescription)
                    texture_img.save(osp.join(output_path, "rooms", room.room_id, "{surface}_texture.png".format(surface=surface)))
                    f.write("<img src='{texture_path}' />".format(texture_path=osp.join("rooms", room.room_id,
                                                                                        "{surface}_texture.png".format(surface=surface))))
                    f.write("<br\n>")
                    observed = ""
                    if texture_img.source.observed is not None:
                        if texture_img.source.observed:
                            observed = "Observed"
                        else:
                            observed = "Unobserved"

                    f.write("<span>{observed} {generator}</span>".format(observed=observed, generator=texture_img.source.name))
                f.write("</td>\n")

            # Preview photos
            f.write("<td>\n")
            for photo in gt_room.photos:
                if not osp.exists(osp.join(output_path, "photos", photo + ".jpg")):
                    os.symlink(osp.join(photos_path, photo + ".jpg"), osp.join(output_path, "photos", photo + ".jpg"))

                border_color = "green"
                if photo not in room.photos:
                    border_color = "red"
                f.write("<img style='border: solid 5px {border_color};' src='{image_path}'/>".format(image_path=osp.join("photos", photo + ".jpg"),
                                                                                                     border_color=border_color))
            f.write("</td>")

            f.write("</tr>\n")
        f.write("</table>\n")
        f.write("</body>\n")
        f.write("</html>\n")


if __name__ == "__main__":
    """
    Generate result pages that preview texture synthesized houses alongside their photos. 
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Preview texture predicted houses")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Path to save result pages")

    parser.add_argument("archs_or_scenes_path", help="Path to generated arch.json files or scene.json files"
                                                     "Usually './data/processed/gnn_prop/[split]/drop_[drop_fraction]/archs'.")
    parser.add_argument("--textures-path", default=None, help="Path to 'tileable_texture_crops' directory containing predicted textures.")
    parser.add_argument("photos_path", help="Path to directory containing photos")
    parser.add_argument("split", help="train/val/test")
    parser.add_argument("drop_fraction", help="Specify photo unobserve probability.")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    input_path = args.archs_or_scenes_path
    photos_path = args.photos_path
    split = args.split
    textures_path = args.textures_path
    conf.drop_fraction = args.drop_fraction

    house_keys = conf.get_data_list(split)
    house_keys = sorted(house_keys)

    # We identify available photos from houses.
    # GT houses contain all the photos.
    gt_houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                     house_key="{house_key}"),
                             photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                                drop_fraction="0.0",  # No drop
                                                                                                house_key="{house_key}"))

    # Houses doesn't contain dropped photos.
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))

    with open(osp.join(output_path, "preview.html"), "w") as f:
        f.write("<html>\n")
        f.write(HTML_HEADER)
        f.write("<body>\n")
        f.write("<table>\n")
        f.write("<tr>\n")
        columns = ["#", "house_id", "preview"]
        f.write("<tr>%s</tr>\n" % ("".join(["<th>%s</th>" % a for a in columns])))

        for i, (house_key) in enumerate(house_keys):
            logging.info("[%d/%d] Previewing House %s" % (i, len(house_keys), house_key))
            f.write("<tr>\n")
            f.write(f"<td>{i}</td>\n")
            f.write("<td><a href='{link}'>{house_key}</a></td>\n".format(house_key=house_key, link=osp.join(house_key, "report.html")))
            house_json_path = None
            house_textures_path = None
            if input_path:
                if osp.exists(osp.join(input_path, house_key + ".arch.json")):
                    house_json_path = osp.join(input_path, house_key + ".arch.json")
                elif osp.exists(osp.join(input_path, house_key + ".scene.json")):
                    house_json_path = osp.join(input_path, house_key + ".scene.json")
                else:
                    logging.warning("Not Found: {path}".format(path=osp.join(input_path, house_key + ".house.json")))

            if textures_path:
                # Load textures
                load_house_crops(conf, houses[house_key], osp.join(textures_path, house_key))

            preview_house(houses[house_key], gt_houses[house_key], house_json_path, photos_path, osp.join(output_path, house_key))

            f.write("<td><img style='width:300px' src='{preview_path}'/></td>\n".format(preview_path=osp.join(house_key, "render.png")))
            f.write("</tr>\n")
            f.flush()

        f.write("</table>\n")
        f.write("</body>\n")
        f.write("</html>\n")
