from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
import logging


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
