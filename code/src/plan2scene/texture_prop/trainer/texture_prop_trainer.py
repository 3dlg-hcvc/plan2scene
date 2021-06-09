import logging

from torch_geometric.data import DataLoader
import os
from config_parser import Config
from plan2scene.common.house_parser import load_houses_with_embeddings, load_houses_with_textures, parse_houses
from plan2scene.common.image_description import ImageSource
from plan2scene.common.trainer.epoch_summary import EpochSummary
from plan2scene.common.trainer.save_reason import SaveReason
from plan2scene.crop_select.util import fill_textures
from plan2scene.evaluation.evaluator import evaluate
from plan2scene.evaluation.matchers import PairedMatcher, UnpairedMatcher
from plan2scene.evaluation.metric_impl.substance_classifier.classifier import SubstanceClassifier
from plan2scene.evaluation.metrics import FreqHistL1, HSLHistL1, ClassificationError, TileabilityMean
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
from plan2scene.texture_prop.houses_dataset import HouseDataset
from plan2scene.texture_prop.trainer.abstract_trainer import AbstractTrainer
from plan2scene.texture_prop.trainer.metric_description import MetricDescription, MetricResult
from plan2scene.config_manager import ConfigManager
import multiprocessing
import os.path as osp
import torch

from plan2scene.texture_prop.utils import get_graph_generator, get_network, get_crit, get_optim, update_embeddings


class TexturePropTrainer(AbstractTrainer):
    """
    Trainer for the Texture Propagation stage.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize trainer.
        """
        super().__init__(*args, **kwargs)
        self._epoch_counter = multiprocessing.Value("i", 0)  # Used to inform the dataloaders to refresh data.
        self._combined_emb_dim = None
        self._tg_predictor = None

    def _setup_datasets(self) -> None:
        """
        Setup datasets and data loders.
        """
        train_graph_generator = get_graph_generator(self.conf, self.system_conf.train_graph_generator, include_target=True)
        val_graph_generator = get_graph_generator(self.conf, self.system_conf.val_graph_generator,
                                                  include_target=True)  # Used for crude evaluation at every epoch
        nt_graph_generator = get_graph_generator(self.conf, self.system_conf.val_graph_generator,
                                                 include_target=False)  # Used for slow/proper evaluation at specified interval

        self._train_dataset = HouseDataset(load_houses_with_embeddings(self.conf, data_split="train", drop_fraction="0.0",
                                                                       embeddings_path=osp.join(self.conf.data_paths.train_texture_prop_train_data,
                                                                                                "surface_texture_embeddings")),
                                           graph_generator=train_graph_generator, epoch_counter=self._epoch_counter)
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self.system_conf.train.bs, shuffle=self.system_conf.train.shuffle_trainset,
                                            num_workers=self.num_workers)

        self._val_dataset = HouseDataset(load_houses_with_embeddings(self.conf, data_split="val", drop_fraction="0.0",
                                                                     embeddings_path=osp.join(self.conf.data_paths.train_texture_prop_val_data,
                                                                                              "surface_texture_embeddings")),
                                         graph_generator=val_graph_generator)
        self._val_dataloader = DataLoader(self._val_dataset, batch_size=self.system_conf.train.bs)

        self._val_nt_dataset = HouseDataset(load_houses_with_embeddings(self.conf, data_split="val", drop_fraction="0.0",
                                                                        embeddings_path=osp.join(self.conf.data_paths.train_texture_prop_val_data,
                                                                                                 "surface_texture_embeddings")),
                                            graph_generator=nt_graph_generator)
        self._val_nt_dataloader = DataLoader(self._val_nt_dataset, batch_size=self.system_conf.train.bs)

    def _setup_extra(self) -> None:
        """
        Setup additional items such as texture predictor and graph generator.
        """
        self._tg_predictor = TextureGenPredictor(conf=load_conf_eval(config_path=self.conf.texture_gen.texture_synth_conf),
                                                 rgb_median_emb=self.conf.texture_gen.rgb_median_emb)
        self._tg_predictor.load_checkpoint(checkpoint_path=self.conf.texture_gen.checkpoint_path)
        self._combined_emb_dim = self.conf.texture_gen.combined_emb_dim
        if self.system_conf.graph_generator.include_enable_in_target:
            self._combined_emb_dim += 1

    def _setup_metrics(self) -> list:
        """
        Setup metrics used for deep evaluation purpose.
        :return: List of metric descriptions
        """
        return [
            MetricDescription("color", PairedMatcher(HSLHistL1())),
            MetricDescription("subs", PairedMatcher(ClassificationError(SubstanceClassifier(classifier_conf=self.conf.metrics.substance_classifier)))),
            MetricDescription("freq", PairedMatcher(FreqHistL1())),
            MetricDescription("tile", UnpairedMatcher(TileabilityMean(metric_param=self.conf.metrics.tileability_mean_metric))),
        ]

    def _setup_network(self):
        """
        Setup network to be trained.
        :return: Network to be trained.
        """
        return get_network(conf=self.conf, network_arch=self.system_conf.network_arch)

    def _setup_crit(self):
        """
        Setup the loss function.
        :return: Loss function.
        """
        return get_crit(self.conf, self.system_conf.train)

    def _setup_optim(self):
        """
        Setup the optimizier.
        :return: Optimizer.
        """
        return get_optim(self.conf, self.system_conf.train, self.net.parameters())

    def _train_epoch(self, epoch: int) -> EpochSummary:
        """
        Train for an epoch.
        :param epoch: Epoch index.
        :return: Evaluation summary for the epoch on the train set.
        """
        self._epoch_counter.value = epoch
        # Eval summary setup
        epoch_summary = EpochSummary(epoch_loss=0.0, epoch_entry_count=0, epoch_batch_count=0)
        for i, batch in enumerate(self.train_dataloader):
            self.optim.zero_grad()

            output = self.net(batch.to(self.device))
            mask_repeated = batch.y_mask.unsqueeze(-1).repeat([1, 1, self._combined_emb_dim])
            loss = self.crit(output[mask_repeated], batch.y[mask_repeated])

            loss.backward()
            self.optim.step()

            epoch_summary.epoch_loss += loss.item()
            epoch_summary.epoch_entry_count += batch.x.shape[0]
            epoch_summary.epoch_batch_count += 1

        return epoch_summary

    def _eval_epoch(self, epoch: int) -> EpochSummary:
        """
        Evaluate the current model on validation set.
        :param epoch: Epoch index.
        :return: Evaluation results
        """
        epoch_summary = EpochSummary(epoch_loss=0.0, epoch_entry_count=0, epoch_batch_count=0)

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                output = self.net(batch.to(self.device))
                mask_repeated = batch.y_mask.unsqueeze(-1).repeat([1, 1, self._combined_emb_dim])
                loss = self.crit(output[mask_repeated], batch.y[mask_repeated])
                epoch_summary.epoch_loss += loss.item()
                epoch_summary.epoch_entry_count += batch.x.shape[0]
                epoch_summary.epoch_batch_count += 1

        return epoch_summary

    def _deep_eval_epoch(self, epoch) -> list:
        """
        Slow/accurate evaluation by synthesizing final texture.
        :param epoch: Epoch number.
        :return: List of MetricResult items.
        """
        # Load untextured houses
        house_keys = self.conf.get_data_list("val")
        pred_houses = parse_houses(self.conf, house_keys, house_path_spec=self.conf.data_paths.arch_path_spec.format(split="val",
                                                                                                                     house_key="{house_key}"),
                                   photoroom_csv_path_spec=self.conf.data_paths.photoroom_path_spec.format(split="val",
                                                                                                           drop_fraction="0.0",
                                                                                                           house_key="{house_key}"))

        # Predict textures for houses
        for i, batch in enumerate(self._val_nt_dataloader):
            output = self.net(batch.to(self.device))
            update_embeddings(self.conf, pred_houses, batch, output)

        fill_textures(self.conf, pred_houses, log=False, predictor=self._tg_predictor, image_source=ImageSource.GNN_PROP, skip_existing_textures=False)

        # Load ground truth
        gt_houses = load_houses_with_textures(self.conf, "val", "0.0", self.conf.data_paths.gt_reference_crops_val)

        # Evaluate
        eval_results = []
        for metric in self.metrics:
            assert isinstance(metric, MetricDescription)
            # logging.info("Evaluating metric: %s" % str(metric.name))
            result = evaluate(self.conf, pred_houses=pred_houses, gt_houses=gt_houses, matcher=metric.evaluator, log=False)
            eval_results.append(MetricResult(metric, result))
        return eval_results

    def _report(self, epoch: int, train_stats: EpochSummary, val_stats: EpochSummary, deep_eval_results: list) -> None:
        """
        Write train progress to the log after completing an epoch.
        :param epoch: Epoch index.
        :param train_stats: Train set epoch evaluation summary.
        :param val_stats: Validation set epoch evaluation summary.
        :param deep_eval_results: Deep evaluation results if a deep evaluation was undertaken.
        """
        # Log detail report if available
        additional = ""
        if deep_eval_results:
            for deep_eval_result in deep_eval_results:
                isinstance(deep_eval_result, MetricResult)
                additional += "%s: %.7f [%.7f/%d]\t" % (
                    str(deep_eval_result.metric),
                    deep_eval_result.eval_result.total_texture_loss / deep_eval_result.eval_result.surface_count,
                    deep_eval_result.eval_result.total_texture_loss, deep_eval_result.eval_result.surface_count
                )
                self.summary_writer.add_scalar("epoch_val_" + str(deep_eval_result.metric),
                                               deep_eval_result.eval_result.total_texture_loss / deep_eval_result.eval_result.surface_count, epoch)

        # Frequent log
        logging.info("[Epoch %d]\t Train Loss: %.7f\t Val Loss: %.7f\t %s " % (epoch,
                                                                                         train_stats.epoch_loss / train_stats.epoch_batch_count,
                                                                                         val_stats.epoch_loss / val_stats.epoch_batch_count,
                                                                                         additional))
        self.summary_writer.add_scalar("epoch_train_loss", train_stats.epoch_loss / train_stats.epoch_batch_count, epoch)
        self.summary_writer.add_scalar("epoch_val_loss", val_stats.epoch_loss / val_stats.epoch_batch_count, epoch)

    def _save_checkpoint(self, epoch: int, reason: SaveReason, train_stats: EpochSummary, val_stats: EpochSummary) -> None:
        """
        Saves a model checkpoint.
        :param epoch: Epoch index.
        :param reason: Save reason.
        :param train_stats: Train set epoch evaluation summary.
        :param val_stats: Validation set epoch evaluation summary.
        """
        save_path = None
        if reason == SaveReason.BEST_MODEL:
            # Save best model
            logging.info("Saving Best Model")
            save_path = osp.join(self.output_path, "best_models",
                                 "best-tex-val-loss-%.5f-epoch-%d.ckpt" % (
                                     val_stats.epoch_loss / val_stats.epoch_batch_count,
                                     epoch))

        elif reason == SaveReason.INTERVAL:
            # logging.info("Saving Checkpoint")
            save_path = osp.join(self.output_path, "checkpoints",
                                 "loss-%.5f-epoch-%d.ckpt" % (
                                     val_stats.epoch_loss / val_stats.epoch_batch_count,
                                     epoch))
        else:
            assert False

        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))

        payload = {
            "model_state_dict": self.net.state_dict(),
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "optimizer_state_dict": self.optim.state_dict()
        }
        torch.save(payload, save_path)
