from plan2scene.common.house_parser import parse_houses, load_house_crops
from plan2scene.common.residence import Room
from plan2scene.config_manager import ConfigManager
import argparse
import os.path as osp
import os
import numpy as np
import uuid
import logging
import shutil
import json
import torch
from pytorch_fid import fid_score
import pathlib


# Method adopted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L244
def calculate_fid_given_files(conf: ConfigManager, gt_files: list, pred_files: list, batch_size: int):
    """
    Compute FID score between two image lists specified.
    :param conf: Config Manager
    :param gt_files: List of ground truth images
    :param pred_files: List of predicted images
    :param batch_size: Batch size used for computation
    :return: FID score.
    """
    device = conf.metrics.fid.device
    dims = conf.metrics.fid.dims

    gt_files = [pathlib.Path(a) for a in gt_files]
    pred_files = [pathlib.Path(a) for a in pred_files]

    if len(pred_files) == 0:
        logging.info("Skipping due to no predictions")
        return torch.tensor(-1.0)

    if len(gt_files) == 0:
        logging.info("Skipping due to no gt")
        return torch.tensor(-1.0)

    block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = fid_score.InceptionV3([block_idx]).to(device)

    m1, s1 = fid_score.calculate_activation_statistics(gt_files, model, batch_size,
                                                       dims, device)
    m2, s2 = fid_score.calculate_activation_statistics(pred_files, model, batch_size,
                                                       dims, device)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def compute_FID_for_houses(conf: ConfigManager, houses: dict, gt_houses: dict, allowed_room_types: list = None, allowed_surface_types: list = None,
                           log: bool = False, multiprop_count: int = 0) -> dict:
    """
    Compute FID score given houses with ground truth surface crops and predicted textures.
    Computation is restricted to the surfaces that match the specified criteria.
    :param conf: Config Manager
    :param houses: Houses with predicted textures
    :param gt_houses: Houses with ground truth crops assigned.
    :param allowed_room_types: Room types included in the computation.
    :param allowed_surface_types: Surface types included in the computation.
    :param log: Specify true to log.
    :param multiprop_count: Texture synthesis repeat count per surface.
    :return: FID results
    """
    # FID library we use require images to be saved in the disk.
    temp_location = osp.abspath("./tmp/fid")
    random_key = str(uuid.uuid4()).replace("-", "")
    temp_location = osp.join(temp_location, random_key)

    if not osp.exists(temp_location):
        os.makedirs(temp_location)

    gt_crop_paths = []
    pred_crop_paths = []

    # Save all prediction crops
    for house_key, house in houses.items():
        gt_house = gt_houses[house_key]
        os.mkdir(osp.join(temp_location, house_key))
        for room_index, room in house.rooms.items():
            gt_room = gt_house.rooms[room_index]
            assert isinstance(room, Room)
            assert isinstance(gt_room, Room)

            os.mkdir(osp.join(temp_location, house_key, str(room_index)))

            if allowed_room_types is None or len(set(room.types).intersection(set(allowed_room_types))) > 0:
                for surface in conf.surfaces:
                    if allowed_surface_types is None or surface in allowed_surface_types:
                        prediction_crops = room.surface_textures[surface]
                        gt_crops = gt_room.surface_textures[surface]

                        os.mkdir(osp.join(temp_location, house_key, str(room_index), surface))

                        # Save GT Crop
                        for gt_index, (gt_key, gt_crop) in enumerate(gt_crops.items()):
                            gt_crop.image.save(osp.join(temp_location, house_key, str(room_index), surface, f"gt_{gt_index}.png"))
                            gt_crop_paths.append(osp.join(temp_location, house_key, str(room_index), surface, f"gt_{gt_index}.png"))

                        # Save Prediction Crop
                        if multiprop_count == 0:
                            if "prop" in prediction_crops:
                                prediction_crops["prop"].image.save(osp.join(temp_location, house_key, str(room_index), surface, "pred.png"))
                                pred_crop_paths.append(osp.join(temp_location, house_key, str(room_index), surface, "pred.png"))
                        else:
                            for prop_index in range(multiprop_count):
                                if f"prop_{prop_index}" in prediction_crops:
                                    prediction_crops[f"prop_{prop_index}"].image.save(
                                        osp.join(temp_location, house_key, str(room_index), surface, f"pred_{prop_index}.png"))
                                    pred_crop_paths.append(osp.join(temp_location, house_key, str(room_index), surface, f"pred_{prop_index}.png"))

    if log:
        logging.info("Method Crop Count: %d" % (len(pred_crop_paths)))
        logging.info("GT Crop Count: %d" % (len(gt_crop_paths)))

    # Check whether we have sufficient GT crops.
    if len(gt_crop_paths) < conf.metrics.fid.minimum_crop_count / 2:
        logging.warning("Insufficient GT crops. You have only %d crops." % (len(gt_crop_paths)))
    elif len(gt_crop_paths) < conf.metrics.fid.minimum_crop_count:
        logging.warning("Insufficient GT crops. Sampled %d more crops with replacement." % (conf.metrics.fid.minimum_crop_count - len(gt_crop_paths)))
        new_gt_crop_paths = np.random.choice(gt_crop_paths, (conf.metrics.fid.minimum_crop_count - len(gt_crop_paths)))
        gt_crop_paths.extend(new_gt_crop_paths)

    # Check whether we have sufficient predictions
    if len(pred_crop_paths) < conf.metrics.fid.minimum_crop_count:
        logging.warning("Insufficient prediction crops. You have %d crops." % (len(pred_crop_paths)))

    fid_score = calculate_fid_given_files(conf, gt_files=gt_crop_paths, pred_files=pred_crop_paths, batch_size=conf.metrics.fid.batch_size)

    # when done, delete
    shutil.rmtree(temp_location)
    if log:
        logging.info("FID: %.5f [%d/%d]" % (fid_score, len(pred_crop_paths), len(gt_crop_paths)))
    return {
        "score": fid_score.item(),
        "pred_crop_count": len(pred_crop_paths),
        "gt_crop_count": len(gt_crop_paths)
    }


def compute_FID_scores(conf: ConfigManager, gt_houses: dict, pred_houses: dict, log: bool = False, multiprop_count: int = 0) -> dict:
    """
    Compute FID scores between textures assigned to pred_houses and gt_houses. We consider various criteria such as all surfaces FID score,
    room type conditioned FID score, surface type conditioned FID score.
    :param conf: Config manager.
    :param gt_houses: Houses having ground truth reference crops assigned.
    :param pred_houses: Houses having predicted textures.
    :param log: Specify true to log.
    :param multiprop_count: Texture synthesis repeat count per surface.
    :return: FID report
    """
    # All Surfaces
    if log:
        logging.info("For all surfaces:")
    all_surfaces_fid_score = compute_FID_for_houses(conf, pred_houses, gt_houses, allowed_room_types=None,
                                                    allowed_surface_types=None, log=log, multiprop_count=multiprop_count)
    # For different surface types
    surface_typed_fid_scores = {}
    for surface_type in conf.surfaces:
        if log:
            logging.info(f"For {surface_type}:")
        surface_typed_fid_scores[surface_type] = compute_FID_for_houses(conf, pred_houses, gt_houses,
                                                                        allowed_room_types=None, allowed_surface_types=[surface_type], log=log,
                                                                        multiprop_count=multiprop_count)
    # For different room types
    room_typed_fid_scores = {}
    for room_type in conf.room_types:
        if log:
            logging.info(f"For {room_type}:")
        room_typed_fid_scores[room_type] = compute_FID_for_houses(conf, pred_houses, gt_houses,
                                                                  allowed_room_types=[room_type], allowed_surface_types=None, log=log,
                                                                  multiprop_count=multiprop_count)

    if log:
        # Log results
        logging.info("")
        logging.info("Results")
        logging.info("MultiProp Count: %d" % (multiprop_count))
        for surface_type in conf.surfaces:
            logging.info("%s FID: %.6f\t [Pred: %d\t GT: %d]" % (

                surface_type, surface_typed_fid_scores[surface_type]["score"], surface_typed_fid_scores[surface_type]["pred_crop_count"],
                surface_typed_fid_scores[surface_type]["gt_crop_count"]))

        for room_type in conf.room_types:
            logging.info("%s FID: %.6f\t [Pred: %d\t GT: %d]" % (
                room_type, room_typed_fid_scores[room_type]["score"], room_typed_fid_scores[room_type]["pred_crop_count"],
                room_typed_fid_scores[room_type]["gt_crop_count"]))

        logging.info("All surfaces FID: %.6f\t [Pred: %d\t GT: %d]" % (
            all_surfaces_fid_score["score"], all_surfaces_fid_score["pred_crop_count"], all_surfaces_fid_score["gt_crop_count"]))

    return {
        "multiprop_count": multiprop_count,
        "all_surfaces_fid": all_surfaces_fid_score,
        "room_typed_fid_scores": {k: v for k, v in room_typed_fid_scores.items()},
        "surface_typed_fid_scores": {k: v for k, v in surface_typed_fid_scores.items()}
    }


def process(conf: ConfigManager, output_path: str, pred_textures_path: str, gt_crops_path: str, split: str, exclude_prior_predictions_path: str,
            restrict_prior_predictions_path: str, log: bool, multiprop_count: int) -> None:
    """
    Compute FID scores for specified surfaces of a house.
    :param conf: Config Manager
    :param output_path: Path to output results directory.
    :param pred_textures_path: Path containing predicted textures for houses.
    :param gt_crops_path: Path containing rectified crops assigned to houses.
    :param split: Val/test
    :param exclude_prior_predictions_path: Surfaces that has texture predictions at this path are excluded from the evaluation.
    :param restrict_prior_predictions_path: Evaluation is exclusive to the surface that has a texture prediction at this path.
    :param log: Specify true to log.
    :param multiprop_count: Texture synthesis repeat count per surface.
    """
    # Load houses
    house_keys = conf.get_data_list(split)
    gt_houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                     house_key="{house_key}"),
                             photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                                drop_fraction=conf.drop_fraction,
                                                                                                house_key="{house_key}"))

    pred_houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                       house_key="{house_key}"),
                               photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                                  drop_fraction=conf.drop_fraction,
                                                                                                  house_key="{house_key}"))

    # Load ground truth crops
    for i, (house_key, house) in enumerate(gt_houses.items()):
        if log:
            logging.info("[%d/%d] Loading GT House %s" % (i, len(gt_houses), house_key))
        load_house_crops(conf, house, osp.join(gt_crops_path, house_key))

    # Load predictions
    for i, (house_key, house) in enumerate(pred_houses.items()):
        if log:
            logging.info("[%d/%d] Loading Prediction House %s" % (i, len(pred_houses), house_key))
        exclude_prior_predictions_house_path = None
        restrict_prior_predictions_house_path = None
        if restrict_prior_predictions_path is not None:
            restrict_prior_predictions_house_path = osp.join(restrict_prior_predictions_path, house_key)
        if exclude_prior_predictions_path is not None:
            exclude_prior_predictions_house_path = osp.join(exclude_prior_predictions_path, house_key)

        load_house_crops(conf, house,
                         osp.join(pred_textures_path, house_key), exclude_prior_path=exclude_prior_predictions_house_path,
                         restrict_prior_path=restrict_prior_predictions_house_path)

    results = compute_FID_scores(conf=conf, pred_houses=pred_houses, gt_houses=gt_houses, log=log,
                                 multiprop_count=multiprop_count)

    if not osp.exists(output_path):
        os.makedirs(output_path)
    with open(osp.join(output_path, "fid_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Saved {path}".format(path=osp.join(output_path, "fid_results.json")))


if __name__ == "__main__":
    """
    Use this metric to compute FID metrics.
    """

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Compute FID metrics.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Path to save fid_results.json. Usually './data/processed/gnn_prop/[split]/drop_0.0/fid_results")
    parser.add_argument("pred_textures_path",
                        help="Path to predicted textures. Usually './data/processed/gnn_prop/[split]/drop_0.0/tileable_texture_crops'.")
    parser.add_argument("gt_crops_path",
                        help="Path to rectified crops extracted from photos. Usually './data/processed/texture_gen/[split]/drop_0.0/texture_crops'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("--exclude-prior-predictions",
                        default=None,
                        help="Specify this argument to evaluate unobserved surfaces. "
                             "Provide the save path of texture crops for observed surfaces so those surfaces can be excluded from the evaluation."
                             "Usually './data/processed/vgg_crop_select/[split]/drop_0.0/texture_crops")
    parser.add_argument("--restrict-to-prior-predictions",
                        default=None,
                        help="Specify this argument to evaluate observed surfaces. "
                             "Provide the save path of texture crops for observed surfaces so we can identify observed surfaces."
                             "Usually './data/processed/vgg_crop_select/[split]/drop_0.0/texture_crops")
    parser.add_argument("--all-cases", default=False, action="store_true", help="Separately evaluate observed, unobserved and all surfaces.")
    parser.add_argument("--prior-predictions",
                        help="This argument is used if all-cases tag is specified. Provide the save path of texture crops for observed surfaces. "
                             "Usually './data/processed/vgg_crop_select/[split]/drop_0.0/texture_crops")
    parser.add_argument("--multiprop", default=0,
                        help="Number of multiple synthesis available per surface. One must provide at-least 2048 total texture predictions to accurately compute FID.")
    args = parser.parse_args()

    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    gt_crops_path = args.gt_crops_path
    pred_textures_path = args.pred_textures_path
    split = args.split
    ex_prior_predictions = args.exclude_prior_predictions
    res_prior_predictions = args.restrict_to_prior_predictions
    all_cases = args.all_cases
    prior_predictions = args.prior_predictions
    multiprop_count = int(args.multiprop)

    if all_cases:
        assert prior_predictions is not None and ex_prior_predictions is None and res_prior_predictions is None
    else:
        assert prior_predictions is None

    if not all_cases:
        process(conf, output_path, pred_textures_path, gt_crops_path, split, log=True, exclude_prior_predictions_path=ex_prior_predictions,
                restrict_prior_predictions_path=res_prior_predictions, multiprop_count=multiprop_count)
    else:
        logging.info("Computing FID for all surfaces.")
        logging.info(".................................")
        process(conf, osp.join(output_path, "all_surfaces"), pred_textures_path, gt_crops_path, split, log=True, exclude_prior_predictions_path=None,
                restrict_prior_predictions_path=None, multiprop_count=multiprop_count)

        logging.info("Computing FID for observed surfaces.")
        logging.info(".................................")
        process(conf, osp.join(output_path, "observed_surfaces"), pred_textures_path, gt_crops_path, split, log=True, exclude_prior_predictions_path=None,
                restrict_prior_predictions_path=prior_predictions, multiprop_count=multiprop_count)

        logging.info("Computing FID for unobserved surfaces.")
        logging.info(".................................")
        process(conf, osp.join(output_path, "unobserved_surfaces"), pred_textures_path, gt_crops_path, split, log=True,
                exclude_prior_predictions_path=prior_predictions, restrict_prior_predictions_path=None, multiprop_count=multiprop_count)
