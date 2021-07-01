from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
import logging
import os.path as osp

from plan2scene.utils.io import load_image


def fill_texture_embeddings(conf: ConfigManager, house: House, predictor: TextureGenPredictor) -> None:
    """
    Compute surface texture embeddings of a house
    :param conf: Config Manager
    :param house: House processed
    :param predictor: Predictor with loaded checkpoint
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        for photo in room.photos:
            for surface in conf.surfaces:
                surface_instances = [i for i in range(conf.texture_gen.masks_per_surface[surface])]
                for surface_instance in surface_instances:
                    for crop_instance in range(conf.texture_gen.crops_per_mask):
                        candidate_key = "%s_%d_crop%d" % (photo, surface_instance, crop_instance)
                        if osp.exists(osp.join(conf.data_paths.rectified_crops_path, surface, candidate_key + ".png")):
                            image = load_image(
                                osp.join(conf.data_paths.rectified_crops_path, surface, candidate_key + ".png"))

                            emb, loss = predictor.predict_embs([image])
                            room.surface_textures[surface][candidate_key] = ImageDescription(image, ImageSource.NEURAL_SYNTH)
                            room.surface_embeddings[surface][candidate_key] = emb
                            room.surface_losses[surface][candidate_key] = loss


def fill_house_textures(conf: ConfigManager, house: House, image_source: ImageSource, skip_existing_textures: bool, key="prop",
                        predictor: TextureGenPredictor = None) -> None:
    """
    Synthesize textures for a house using the assigned texture embeddings.
    :param conf: Config Manager
    :param house: House to populate textures
    :param key: Key of candidate texture embeddings.
    :param image_source: Generator of the images
    :param predictor: Predictor used to synthesize textures
    :param skip_existing_textures: Do no synthesize if a texture already exist
    """
    if predictor is None:
        predictor = TextureGenPredictor(
            conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
            rgb_median_emb=conf.texture_gen.rgb_median_emb)
        predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        for surface in room.surface_embeddings:
            if key in room.surface_embeddings[surface]:
                if skip_existing_textures and key in room.surface_textures[surface]:
                    continue
                generated_crops, substance_names, extra = predictor.predict_textures(
                    combined_embs=[room.surface_embeddings[surface][key]],
                    multiplier=conf.texture_gen.output_multiplier)
                room.surface_textures[surface][key] = ImageDescription(generated_crops[0], image_source)


def fill_textures(conf: ConfigManager, houses: dict, image_source: ImageSource, skip_existing_textures: bool, key: str = "prop", log: bool = True,
                  predictor: TextureGenPredictor = None) -> None:
    """
    Synthesize textures for houses using the assigned texture embeddings.
    :param conf: Config manager
    :param houses: Dictionary of houses.
    :param image_source: Image source specified to the synthesized textures
    :param skip_existing_textures: Specify true to keep existing textures. Specify false to replace existing textures with new textures.
    :param key: Key of embeddings used to synthesize textures.
    :param log: Set true to enable logging.
    :param predictor: Predictor used to synthesize textures.
    """
    if predictor is None:
        predictor = TextureGenPredictor(
            conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
            rgb_median_emb=conf.texture_gen.rgb_median_emb)
        predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    for i, (house_key, house) in enumerate(houses.items()):
        if log:
            logging.info("[%d/%d] Generating Textures %s" % (i, len(houses), house_key))
        fill_house_textures(conf, house, skip_existing_textures=skip_existing_textures, key=key, predictor=predictor, image_source=image_source)


def get_least_key(kv):
    """
    Given a dictionary, returns the key with minimum value.
    :param kv: Dictionary considered.
    :return: Key with the minimum value.
    """
    min_k = None
    min_v = None
    for k, v in kv.items():
        if min_v is None or v.item() < min_v:
            min_k = k
            min_v = v.item()

    return min_k


def vgg_crop_select(conf: ConfigManager, house: House, predictor: TextureGenPredictor) -> None:
    """
    Assigns the least VGG loss crop for each surface of the house.
    :param conf: ConfigManager
    :param house: House to update
    :param predictor: Predictor used to synthesize textures
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        # Calculate the least VGG loss embeddings
        for surface in room.surface_embeddings:
            least_key = get_least_key(room.surface_losses[surface])
            if least_key is not None:
                room.surface_embeddings[surface] = {"prop": room.surface_embeddings[surface][least_key]}
                room.surface_losses[surface] = {"prop": room.surface_losses[surface][least_key]}
            else:
                room.surface_embeddings[surface] = {}
                room.surface_losses[surface] = {}

    fill_textures(conf, {house.house_key: house}, predictor=predictor, log=False, image_source=ImageSource.VGG_CROP_SELECT, skip_existing_textures=False)
