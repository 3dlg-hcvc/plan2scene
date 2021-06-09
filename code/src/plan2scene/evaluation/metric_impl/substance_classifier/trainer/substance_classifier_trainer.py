from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_parser import Config
from plan2scene.common.trainer.save_reason import SaveReason
from plan2scene.config_manager import ConfigManager
from plan2scene.evaluation.metric_impl.substance_classifier.dataset import SubstanceCropDataset
from plan2scene.evaluation.metric_impl.substance_classifier.trainer.abstract_trainer import AbstractTrainer

import os.path as osp
import os
import torch
from plan2scene.evaluation.metric_impl.substance_classifier.trainer.epoch_summary import SubstanceClassifierEpochSummary
from plan2scene.evaluation.metric_impl.substance_classifier.util import get_model, get_crit, get_optim, save_checkpoint
import logging

RUN_SIZE = 100


class SubstanceClassifierTrainer(AbstractTrainer):
    """
    Trainer used to train the substance classifier of the SUBS metric.
    """

    def __init__(self, conf: ConfigManager, system_conf: Config, output_path: str, summary_writer: SummaryWriter, save_model_interval: int,
                 preview_results: bool):
        """
        Initializes trainer.
        :param conf: Config manager
        :param system_conf: Configuration of the trained model.
        :param output_path: Output path to save train outputs.
        :param summary_writer: Summary writer used for logging.
        :param save_model_interval: Frequency of saving checkpoints.
        :param preview_results: Specify true to save previews of correct predictions and mistakes.
        """
        super().__init__(conf, system_conf, output_path=output_path, summary_writer=summary_writer, save_model_interval=save_model_interval)
        self.should_preview_results = preview_results

    def _setup_network(self):
        return get_model(self.system_conf.arch, self.system_conf.substances)

    def _setup_crit(self):
        return get_crit(self.system_conf.train.crit, self.system_conf.train.weight_loss_classes, self.train_dataset, self.system_conf.substances)

    def _setup_optim(self):
        return get_optim(self.system_conf.train.optim, lr=self.system_conf.train.lr, params=self.net.parameters())

    def _preview_results(self, out_path, preview_entries, epoch, split, correct_count, total_count) -> None:
        """
        Save predictions to disk.
        :param out_path: Path to save results.
        :param preview_entries: Entries to save.
        :param epoch: Epoch
        :param split: train or val
        :param correct_count: Correct predictions count.
        :param total_count: Total predictions count.
        """
        for source in ["textures", "os"]:
            output_path = osp.join(out_path, source)
            if not osp.exists(output_path):
                os.makedirs(output_path)
            if not osp.exists(osp.join(output_path, "crops")):
                os.mkdir(osp.join(output_path, "crops"))

            with open(osp.join(output_path, "index.htm"), "w") as f:
                f.write("<h1>%s: Epoch %d</h1>" % (split, epoch))
                f.write("<h2>Accuracy: %.5f [%d/%d]</h2>" % (float(correct_count) / total_count, correct_count, total_count))
                for i, mistake in enumerate(preview_entries):
                    if mistake["source"] == source:
                        file_path = mistake["file_path"]
                        if osp.exists(osp.join(output_path, "crops", "%d.png" % i)):
                            os.remove(osp.join(output_path, "crops", "%d.png" % i))
                        os.symlink(osp.abspath(file_path), osp.join(output_path, "crops", "%d.png" % i))
                        pred = mistake["pred"]
                        truth = mistake["truth"]
                        f.write("<div style='float:left; margin:5px;'><img src='%s'/><br><small>%s</small><br><small>%s</small></div>" % (
                            "crops/%d.png" % (i), "pred_" + pred, "truth_" + truth))

    def _setup_dataset(self):
        self._train_dataset = SubstanceCropDataset(os_dataset_path=osp.join(self.system_conf.datasets.os, "train"),
                                                   texture_dataset_path=osp.join(self.system_conf.datasets.textures, "train"),
                                                   label_mapping=self.system_conf.datasets.label_mapping.__dict__,
                                                   substances=self.system_conf.substances,
                                                   train=True)
        self._train_dataloader = DataLoader(self.train_dataset, batch_size=self.system_conf.train.bs, shuffle=self.system_conf.train.shuffle,
                                            drop_last=self.system_conf.train.drop_last)

        self._val_dataset = SubstanceCropDataset(os_dataset_path=osp.join(self.system_conf.datasets.os, "val"),
                                                 texture_dataset_path=osp.join(self.system_conf.datasets.textures, "val"),
                                                 label_mapping=self.system_conf.datasets.label_mapping.__dict__, substances=self.system_conf.substances,
                                                 train=False)
        self._val_dataloader = DataLoader(self.val_dataset, batch_size=self.system_conf.train.val.bs, shuffle=self.system_conf.train.val.shuffle,
                                          drop_last=self.system_conf.train.val.drop_last)

    def _train_epoch(self, epoch: int) -> SubstanceClassifierEpochSummary:
        """
        Train for an epoch.
        :param epoch: Epoch number
        :return: Epoch summary.
        """
        substances = self.system_conf.substances
        epoch_metrics = SubstanceClassifierEpochSummary(epoch_loss=0.0,
                                                        passed_count=0,
                                                        epoch_entry_count=0,
                                                        epoch_batch_count=0,
                                                        mistakes=[], correct_predictions=[],
                                                        per_substance_count_map={k: 0 for k in (substances)},
                                                        per_substance_correct_count_map={k: 0 for k in (substances)})
        running_loss = 0

        for batch_id, (input_tensor, target, meta) in enumerate(self.train_dataloader):
            input_tensor = input_tensor.to(self.device)
            target = target.to(self.device)

            self.optim.zero_grad()
            outputs = self.net(input_tensor)
            loss = self.crit(outputs, target)
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            epoch_metrics.epoch_loss += loss.item() * self.system_conf.train.bs / input_tensor.shape[0]
            epoch_metrics.epoch_batch_count += 1

            _, preds = torch.max(outputs, dim=1)
            matched = preds == target

            epoch_metrics.passed_count += torch.sum(matched, dim=0).item()
            epoch_metrics.epoch_entry_count += input_tensor.shape[0]

            if batch_id % RUN_SIZE == RUN_SIZE - 1:
                logging.info("[%d, %d] Loss: %.5f" % (epoch, batch_id + 1, running_loss / RUN_SIZE))
                running_loss = 0

            for i, substance in enumerate(meta["substance"]):
                epoch_metrics.per_substance_count_map[substance] += 1
                if matched[i]:
                    epoch_metrics.per_substance_correct_count_map[substance] += 1
                    epoch_metrics.correct_predictions.append({
                        "file_path": meta["file_path"][i],
                        "pred": substances[preds[i]],
                        "truth": meta["substance"][i],
                        "source": meta["source"][i]
                    })
                else:
                    epoch_metrics.mistakes.append({
                        "file_path": meta["file_path"][i],
                        "pred": substances[preds[i]],
                        "truth": meta["substance"][i],
                        "source": meta["source"][i]
                    })

            for i, source in enumerate(meta["source"]):
                # Record correct predictions and predictions per source (opensurfaces vs. textures dataset).
                if source not in epoch_metrics.per_substance_count_map:
                    epoch_metrics.per_substance_count_map[source] = 0
                    epoch_metrics.per_substance_correct_count_map[source] = 0

                substance = meta["substance"][i]
                if source + "_" + substance not in epoch_metrics.per_substance_count_map:
                    epoch_metrics.per_substance_count_map[source + "_" + substance] = 0
                    epoch_metrics.per_substance_correct_count_map[source + "_" + substance] = 0

                epoch_metrics.per_substance_count_map[source] += 1
                epoch_metrics.per_substance_count_map[source + "_" + substance] += 1

                if matched[i]:
                    epoch_metrics.per_substance_correct_count_map[source] += 1
                    epoch_metrics.per_substance_correct_count_map[source + "_" + substance] += 1
        return epoch_metrics

    def _eval_epoch(self, epoch: int) -> SubstanceClassifierEpochSummary:
        """
        Evaluate the model on validation set.
        :param epoch: Epoch.
        :return: Eval epoch summary.
        """
        substances = self.system_conf.substances

        epoch_metrics = SubstanceClassifierEpochSummary(epoch_loss=0.0,
                                                        passed_count=0,
                                                        epoch_entry_count=0,
                                                        epoch_batch_count=0,
                                                        mistakes=[], correct_predictions=[],
                                                        per_substance_count_map={k: 0 for k in (substances)},
                                                        per_substance_correct_count_map={k: 0 for k in (substances)})

        with torch.no_grad():
            for batch_id, (input_tensor, target, meta) in enumerate(self.val_dataloader):
                input_tensor = input_tensor.to(self.device)
                target = target.to(self.device)

                outputs = self.net(input_tensor)
                _, preds = torch.max(outputs, dim=1)
                matched = preds == target

                val_loss = self.crit(outputs, target).item() / input_tensor.shape[0] * self.system_conf.train.val.bs
                epoch_metrics.epoch_loss += val_loss
                epoch_metrics.epoch_batch_count += 1

                epoch_metrics.passed_count += torch.sum(matched, dim=0).item()
                epoch_metrics.epoch_entry_count += input_tensor.shape[0]

                for i, substance in enumerate(meta["substance"]):
                    epoch_metrics.per_substance_count_map[substance] += 1
                    if matched[i]:
                        epoch_metrics.per_substance_correct_count_map[substance] += 1
                        epoch_metrics.correct_predictions.append({
                            "file_path": meta["file_path"][i],
                            "pred": substances[preds[i]],
                            "truth": meta["substance"][i],
                            "source": meta["source"][i]
                        })
                    else:
                        epoch_metrics.mistakes.append({
                            "file_path": meta["file_path"][i],
                            "pred": substances[preds[i]],
                            "truth": meta["substance"][i],
                            "source": meta["source"][i]
                        })

                for i, source in enumerate(meta["source"]):
                    if source not in epoch_metrics.per_substance_count_map:
                        epoch_metrics.per_substance_count_map[source] = 0
                        epoch_metrics.per_substance_correct_count_map[source] = 0

                    substance = meta["substance"][i]
                    if source + "_" + substance not in epoch_metrics.per_substance_count_map:
                        epoch_metrics.per_substance_count_map[source + "_" + substance] = 0
                        epoch_metrics.per_substance_correct_count_map[source + "_" + substance] = 0

                    epoch_metrics.per_substance_count_map[source + "_" + substance] += 1
                    epoch_metrics.per_substance_count_map[source] += 1

                    if matched[i]:
                        epoch_metrics.per_substance_correct_count_map[source] += 1
                        epoch_metrics.per_substance_correct_count_map[source + "_" + substance] += 1

        return epoch_metrics

    def _report(self, epoch: int, train_stats: SubstanceClassifierEpochSummary, val_stats: SubstanceClassifierEpochSummary) -> None:
        """
        Store epoch summary in the disk. (E.g. report to tensorboard.).
        :param epoch: Epoch
        :param train_stats: Validation set summary.
        :param val_stats: Train set summary.
        """
        logging.info("[Epoch %d] train-loss: %.5f\t train-acc: %.4f [%d/%d]\t val-loss: %.5f\t val-acc: %.4f [%d/%d]" % (
            epoch, train_stats.epoch_loss / train_stats.epoch_batch_count,
            float(train_stats.passed_count) / train_stats.epoch_entry_count, train_stats.passed_count, train_stats.epoch_entry_count,
            val_stats.epoch_loss / val_stats.epoch_batch_count,
            float(val_stats.passed_count) / val_stats.epoch_entry_count, val_stats.passed_count, val_stats.epoch_entry_count,
        ))

        self.summary_writer.add_scalar("loss/train", train_stats.epoch_loss / train_stats.epoch_batch_count, epoch)
        self.summary_writer.add_scalar("loss/val", val_stats.epoch_loss / val_stats.epoch_batch_count, epoch)

        self.summary_writer.add_scalar("overall_accuracy/train", float(train_stats.passed_count) / train_stats.epoch_entry_count, epoch)
        self.summary_writer.add_scalar("overall_accuracy/val", float(val_stats.passed_count) / val_stats.epoch_entry_count, epoch)

        if self.should_preview_results:
            # Uncomment to preview correct predictions and mistakes made on the training set
            # self.preview_results(osp.join(self.output_path, "train_mistakes", "epoch_%d" % (epoch)), train_stats.mistakes, epoch, "train",
            #                      train_stats.passed_count, train_stats.epoch_entry_count)
            # self.preview_results(osp.join(self.output_path, "train_correct_predictions", "epoch_%d" % (epoch)), train_stats.correct_predictions, epoch,
            #                      "train", train_stats.passed_count, train_stats.epoch_entry_count)
            self._preview_results(osp.join(self.output_path, "val_mistakes", "epoch_%d" % (epoch)), val_stats.mistakes, epoch, "val",
                                  val_stats.passed_count, val_stats.epoch_entry_count)
            self._preview_results(osp.join(self.output_path, "val_correct_predictions", "epoch_%d" % (epoch)), val_stats.correct_predictions, epoch, "val",
                                  val_stats.passed_count, val_stats.epoch_entry_count)

    def _save_checkpoint(self, epoch: int, save_reason: SaveReason, train_stats: SubstanceClassifierEpochSummary, val_stats: SubstanceClassifierEpochSummary):
        save_checkpoint(self.output_path, self.net, self.optim, save_reason, epoch, val_stats.passed_count, val_stats.epoch_entry_count)
