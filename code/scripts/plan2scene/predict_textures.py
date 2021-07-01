import argparse
import os.path as osp
import os
from arch_parser.parser import parse_scene_json_from_file
from plan2scene.common.house_parser import save_house_crops, save_arch
from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import House
from plan2scene.config_manager import ConfigManager
from plan2scene.crop_select.util import fill_texture_embeddings, vgg_crop_select
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
from plan2scene.texture_prop.gnn_prop import propagate_textures
from plan2scene.texture_prop.predictor import TexturePropPredictor
from plan2scene.utils.io import load_image
import logging

from plan2scene.utils.tile_util import seam_correct_house


def load_house_with_photos(conf: ConfigManager, scene_json_path: str, photoroom_csv_path: str, surface_crops_path: str) -> House:
    """
    Load scene.json file and its associated photos.
    :param conf: Config manager
    :param scene_json_path: Path to scene.json file.
    :param photoroom_csv_path: Path to photoroom.csv file indicating photo to room assignments.
    :param surface_crops_path: Path to directory containing rectified surface crops.
    :return: House
    """
    arch_house = parse_scene_json_from_file(scene_json_path, photoroom_csv_path)
    house = House.from_arch_house(arch_house, surfaces=conf.surfaces)

    # Load associated crops
    for room_index, room in house.rooms.items():
        for photo in room.photos:
            for surface in conf.surfaces:
                surface_instances = [i for i in range(conf.texture_gen.masks_per_surface[surface])]
                for surface_instance in surface_instances:
                    for crop_instance in range(conf.texture_gen.crops_per_mask):
                        candidate_key = "%s_%d_crop%d" % (photo, surface_instance, crop_instance)
                        if osp.exists(osp.join(surface_crops_path, surface, candidate_key + ".png")):
                            image = load_image(
                                osp.join(surface_crops_path, surface, candidate_key + ".png"))
                            room.surface_textures[surface][candidate_key] = ImageDescription(image, ImageSource.NEURAL_SYNTH)
    return house


def process_observed_surfaces(conf: ConfigManager, house: House) -> None:
    """
    Synthesize textures for observed surfaces
    :param conf: Config Manager
    :param house: House processed
    """
    # Load texture synthesis network
    tg_predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    tg_predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    # Compute texture embeddings for observed surfaces (Code adapted from ./code/scripts/preprocessing/fill_room_embeddigs.py)
    fill_texture_embeddings(conf, house, tg_predictor)

    # Synthesize textures for observed surfaces using the most suitable crop identified by VGG textureness score.
    vgg_crop_select(conf, house, tg_predictor)


def process_unobserved_surfaces(conf: ConfigManager, house: House, prop_checkpoint_path) -> None:
    """
    Synthesize textures for unobserved surfaces
    :param conf: Config manager
    :param house: House
    :param prop_checkpoint_path: Path to GNN checkpoint
    """
    # Load GNN and graph generator
    tg_predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    tg_predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    tp_predictor = TexturePropPredictor(conf, conf.texture_prop)
    tp_predictor.load_checkpoint(checkpoint_path=prop_checkpoint_path)

    # Graph dataset
    houses = {house.house_key: house}
    propagate_textures(conf, houses, tg_predictor, tp_predictor, keep_existing_predictions=True, use_train_graph_generator=False, use_val_graph_generator=False)


if __name__ == "__main__":
    """
    End-to-end texture prediction for a house.
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="End-to-end texture prediction for a house")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Path to save the textured scene.json file.")
    parser.add_argument("scene_json_path",
                        help="Path to untextured scene_json file.")
    parser.add_argument("photoroom_csv_path", help="Path to photoroom.csv file describing assignment of photos to rooms.")
    parser.add_argument("surface_crops_path",
                        help="Path to directory containing crops extracted from surfaces. This directory has 3 subdirectories one per each surface type.")
    parser.add_argument("texture_prop", help="Path to configuration file of GNN used for texture propagation.")
    parser.add_argument("texture_prop_checkpoint_path", help="Path to checkpoint file of GNN used for texture propagation.")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    scene_json_path = args.scene_json_path
    photoroom_csv_path = args.photoroom_csv_path
    surface_crops_path = args.surface_crops_path
    texture_prop_checkpoint_path = args.texture_prop_checkpoint_path

    # Processing
    house = load_house_with_photos(conf, scene_json_path, photoroom_csv_path, surface_crops_path)
    logging.info("Loaded house with {rooms} rooms.".format(rooms=len(house.rooms)))
    house.sketch_house().convert("RGB").save(osp.join(output_path, "floorplan_sketch.png"))
    logging.info("")

    # Observed surfaces
    logging.info("Processing observed surfaces.")
    process_observed_surfaces(conf, house)
    logging.info("")

    # Unobserved surfaces
    logging.info("Processing unobserved surfaces.")
    process_unobserved_surfaces(conf, house, texture_prop_checkpoint_path)
    logging.info("")

    # Seam correct textures
    logging.info("Seam correction")
    embark_texture_synthesis_path = conf.seam_correct_config.texture_synthesis_path
    seam_mask_path = conf.seam_correct_config.seam_mask_path
    if osp.exists(embark_texture_synthesis_path) and osp.exists(seam_mask_path):
        seam_correct_house(house, embark_texture_synthesis_path, seam_mask_path)
    else:
        logging.warning("Seam correction stage is skipped due to missing configuration.")

    # Save results
    save_house_crops(house, osp.join(output_path, "texture_predictions"))
    logging.info("Saved texture predictions in '{output_path}/texture_predictions'.".format(output_path=output_path))

    # Save output house
    if not osp.exists(osp.join(output_path, "textured_arch")):
        os.mkdir(osp.join(output_path, "textured_arch"))
    save_arch(conf, house, osp.join(output_path, "textured_arch", house.house_key), texture_both_sides_of_walls=True)
    logging.info(
        "Saved textured architecture to '{output_path}/textured_arch/{house_key}.scene.json'.".format(output_path=output_path, house_key=house.house_key))
