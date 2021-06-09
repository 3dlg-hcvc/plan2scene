#!/bin/python3

from plan2scene.config_manager import ConfigManager
import argparse
import logging
import os.path as osp
import os
from plan2scene.common.house_parser import parse_houses, load_house_crops
from plan2scene.evaluation.evaluator import evaluate
from plan2scene.evaluation.matchers import PairedMatcher, UnpairedMatcher
from plan2scene.evaluation.metric_impl.substance_classifier.classifier import SubstanceClassifier
from plan2scene.evaluation.metrics import HSLHistL1, FreqHistL1, TileabilityMean, ClassificationError

if __name__ == "__main__":
    """
    Evaluate texture predictions.
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Evaluate texture predictions.")
    conf.add_args(parser)
    parser.add_argument("pred_path",
                        help="Path to predicted textures of houses. Usually './data/processed/gnn_prop/[split]/drop_[drop_fraction]/tileable_texture_crops'.")
    parser.add_argument("gt_path",
                        help="Path to ground truth reference crops. Usually './data/processed/gt_reference/[split]/texture_crops'.")
    parser.add_argument("split", help="train/val/test")
    parser.add_argument("--exclude-prior-predictions",
                        default=None,
                        help="Specify this argument to evaluate unobserved surfaces. "
                             "Provide the save path of texture crops for observed surfaces so those surfaces can be excluded from the evaluation."
                             "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]/texture_crops")

    args = parser.parse_args()
    conf.process_args(args)

    gt_path = args.gt_path
    pred_path = args.pred_path
    split = args.split
    exclude_prior_predictions = args.exclude_prior_predictions

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

    for i, (house_key, house) in enumerate(gt_houses.items()):
        logging.info("[%d/%d] Loading GT House %s" % (i, len(gt_houses), house_key))
        exclude_prior_predictions_house_path = None
        if exclude_prior_predictions is not None:
            exclude_prior_predictions_house_path = osp.join(exclude_prior_predictions, house_key)
        load_house_crops(conf, house,
                         osp.join(gt_path, house_key), exclude_prior_path=exclude_prior_predictions_house_path)

    for i, (house_key, house) in enumerate(pred_houses.items()):
        logging.info("[%d/%d] Loading Prediction House %s" % (i, len(pred_houses), house_key))
        exclude_prior_predictions_house_path = None
        if exclude_prior_predictions is not None:
            exclude_prior_predictions_house_path = osp.join(exclude_prior_predictions, house_key)
        load_house_crops(conf, house,
                         osp.join(pred_path, house_key), exclude_prior_path=exclude_prior_predictions_house_path)

    eval_tasks = [PairedMatcher(HSLHistL1()), PairedMatcher(FreqHistL1())]
    if osp.exists(conf.metrics.substance_classifier.checkpoint_path):
        eval_tasks.append(PairedMatcher(ClassificationError(SubstanceClassifier(conf.metrics.substance_classifier))))
    else:
        logging.info("Skipping SUBS metric. Checkpoint not found for substance classifier.")
    eval_tasks.append(UnpairedMatcher(TileabilityMean(conf.metrics.tileability_mean_metric)))
    eval_results = []
    for matcher in eval_tasks:
        logging.info("Evaluating metric: %s" % str(matcher))
        result = evaluate(conf, pred_houses=pred_houses, gt_houses=gt_houses, matcher=matcher)
        eval_results.append(result)

    for matcher, result in zip(eval_tasks, eval_results):
        print("{matcher}: {score:.3f} ({surface_count} surfaces)".format(matcher=str(matcher), score=result.total_texture_loss / result.surface_count,
                                                                         surface_count=result.surface_count))
