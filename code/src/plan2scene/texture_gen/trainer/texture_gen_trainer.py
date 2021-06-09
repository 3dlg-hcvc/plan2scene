from torch.utils.data import DataLoader

from config_parser import Config
import plan2scene.texture_gen.utils.neural_texture_helper as utils_nt
import plan2scene.texture_gen.utils.utils as util
from plan2scene.common.color_description import ColorSpace, Color
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.trainer import train_util
from plan2scene.texture_gen.trainer.abstract_trainer import AbstractTrainer, SaveReason
from plan2scene.texture_gen.trainer.epoch_summary import TextureGenEpochSummary
from plan2scene.texture_gen.trainer.image_dataset import ImageDataset
from plan2scene.texture_gen.utils.hsv_utils import hsv_to_rgb
from plan2scene.texture_gen.utils.io import preview_images, preview_deltas
import torchvision
from PIL import ImageDraw
import os.path as osp
import torch
import logging

# Print train-log every 100 epochs
RUN_SIZE = 100


class TextureGenTrainer(AbstractTrainer):
    """
    Trainer for Modified Neural Texture Synthesis stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style_criterion = None
        self._substance_criterion = None

    def _setup_datasets(self) -> None:
        """
        Setup datasets and data loaders
        """
        logging.info("Train Data: %s" % osp.join(self.train_params.dataset.path, "train"))
        self._train_dataset = ImageDataset(osp.join(self.train_params.dataset.path, "train"),
                                           image_res=(
                                               self.train_params.image.image_res, self.train_params.image.image_res),
                                           resample_count=self.train_params.train.resample_count * self.train_params.train.bs,
                                           scale_factor=self.train_params.image.scale_factor,
                                           substances=self._predictor.substances)
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self.train_params.train.bs, shuffle=True,
                                            num_workers=self.train_params.num_workers,
                                            drop_last=True)

        logging.info("Val Data: %s" % osp.join(self.train_params.dataset.path, "val"))
        self._val_dataset = ImageDataset(osp.join(self.train_params.dataset.path, "val"),
                                         image_res=(
                                             self.train_params.image.image_res, self.train_params.image.image_res),
                                         resample_count=1,
                                         scale_factor=self.train_params.image.scale_factor,
                                         substances=self._predictor.substances)
        self._val_dataloader = DataLoader(self._val_dataset, batch_size=self.train_params.train.bs, shuffle=False,
                                          num_workers=self.train_params.num_workers,
                                          drop_last=False)

    def _setup_predictor(self) -> None:
        """
        Setup predictor
        """
        self._predictor = TextureGenPredictor(conf=self.train_params, rgb_median_emb=False)

    def _setup_loss_function(self) -> None:
        """
        Setup loss function
        """
        self._style_criterion = train_util.get_loss(self.train_params.system.loss_params.style_loss)
        if self.predictor.substances is not None:
            self._substance_criterion = train_util.get_loss(self.train_params.system.loss_params.substance_loss)

        self._vgg_features = utils_nt.VGGFeatures().to(self.device)
        self._gram_matrix = utils_nt.GramMatrix().to(self.device)

    def _setup_optimizer(self) -> None:
        """
        Setup optimizer
        :return:
        """
        self._optimizer = train_util.get_optim(self.train_params.system.optimizer_params,
                                               self.predictor.net.parameters())

    def _train_epoch(self, epoch: int) -> TextureGenEpochSummary:
        """
        Train an epoch
        :param epoch: Epoch index
        :return: Epoch train summary
        """
        # Metrics setup
        epoch_metrics = TextureGenEpochSummary(epoch_loss=0.0,
                                               epoch_style_loss=0.0,
                                               epoch_substance_loss=0.0,
                                               epoch_substance_passed=0,
                                               epoch_entry_count=0,
                                               epoch_batch_count=0)

        running_metrics = Config(dict(loss=0, substance_loss=0, style_loss=0))

        # Iterate data
        for i, batch in enumerate(self.train_dataloader):
            unsigned_images, filenames, sub_targets, unsigned_hsv_images = batch

            sub_targets = sub_targets.to(self.device)
            image_gt = util.unsigned_to_signed(unsigned_images).to(self.device)

            self.optimizer.zero_grad()
            self.predictor.update_seed()

            sample_pos = self.predictor.get_position()

            predictor_result = self.predictor.predict(unsigned_images=unsigned_images,
                                                      unsigned_hsv_images=unsigned_hsv_images,
                                                      sample_pos=sample_pos, train=True)
            image_out = predictor_result.image_out
            substance_out = predictor_result.substance_out

            style_loss = utils_nt.get_loss(image_gt, image_out, self.train_params, self._vgg_features,
                                           self._gram_matrix, self._style_criterion)

            substance_loss = None
            if self.predictor.substances is not None:
                substance_loss = self._substance_criterion(substance_out,
                                                           sub_targets) * self.train_params.system.loss_params.substance_weight
                loss = style_loss + substance_loss
            else:
                loss = style_loss

            loss.backward()
            self.optimizer.step()

            # Compute metrics
            running_metrics.loss += loss.item()
            running_metrics.style_loss += style_loss.item()

            epoch_metrics.epoch_loss += loss.item()
            epoch_metrics.epoch_style_loss += style_loss.item()

            # Compute substance metrics
            if self.predictor.substances is not None:
                running_metrics.substance_loss += substance_loss.item()
                epoch_metrics.epoch_substance_loss += substance_loss.item()

                _, substance_preds = substance_out.max(dim=1)
                substance_pass = (substance_preds == sub_targets).sum(dim=0)

                epoch_metrics.epoch_substance_passed += substance_pass.item()

            epoch_metrics.epoch_entry_count += image_out.shape[0]
            epoch_metrics.epoch_batch_count += 1

            with torch.no_grad():
                # Save first batch of train images to compare with val
                if i == 0:
                    preview_train_imgs = preview_images(
                        torch.cat([util.signed_to_unsigned(image_gt), util.signed_to_unsigned(image_out)], dim=0),
                        self.train_params.train.bs)
                    self.summary_writer.add_image("epoch_train_batch%d_results" % (i + 1), preview_train_imgs,
                                                  global_step=epoch)

            # Print running loss
            if i % RUN_SIZE == RUN_SIZE - 1:
                logging.info("[%d, %d] loss: %.5f\t substance-loss: %.5f\t texture-loss: %.5f" %
                             (epoch, i + 1, running_metrics.loss / RUN_SIZE, running_metrics.substance_loss / RUN_SIZE,
                              running_metrics.style_loss / RUN_SIZE))

                running_metrics.loss = 0
                running_metrics.style_loss = 0
                running_metrics.substance_loss = 0

        return epoch_metrics

    def _preview_base_color(self, val_unsigned_images, extra):
        """
        Create preview image for base color
        :param val_unsigned_images: Synthesized textures
        :param extra: Extra info
        :return: Base color image as a array
        """
        base_color_val = torch.ones_like(val_unsigned_images)

        if extra["base_color"] is not None:
            base_color = extra["base_color"]
            assert isinstance(base_color, Color)
            base_color_val[:, 0, :, :] = base_color.components[0].unsqueeze(1).unsqueeze(
                2).expand(
                [base_color_val.shape[0], base_color_val.shape[2], base_color_val.shape[3]])
            base_color_val[:, 1, :, :] = base_color.components[1].unsqueeze(1).unsqueeze(
                2).expand(
                [base_color_val.shape[0], base_color_val.shape[2], base_color_val.shape[3]])
            base_color_val[:, 2, :, :] = base_color.components[2].unsqueeze(1).unsqueeze(
                2).expand(
                [base_color_val.shape[0], base_color_val.shape[2], base_color_val.shape[3]])
            if base_color.color_space == ColorSpace.HSV:
                base_color_val = hsv_to_rgb(base_color_val)

        return base_color_val

    def _preview_substance_labels(self, val_substance_preds, image_gt_val_for_preview):
        """
        Preview substance labels
        :param val_substance_preds: Substance predictions
        :param image_gt_val_for_preview: Ground truth preview image
        :return:
        """
        for j in range(val_substance_preds.shape[0]):
            img = torchvision.transforms.ToPILImage()(
                util.signed_to_unsigned(image_gt_val_for_preview[j]).cpu())
            img_draw = ImageDraw.Draw(img)
            img_draw.text((0, 0), self.predictor.substances[val_substance_preds[j].item()],
                          fill=(255, 0, 0))
            image_gt_val_for_preview[j] = util.unsigned_to_signed(
                torchvision.transforms.ToTensor()(img)).to(self.device)

    def _update_eval_metrics(self, epoch: int, val_i: int, epoch_metrics: TextureGenEpochSummary, val_unsigned_images,
                             val_sub_targets, image_out_val, substance_out_val, extra) -> None:
        """
        Report epoch evaluations considering a new batch
        :param epoch: Epoch
        :param val_i: Batch idx
        :param epoch_metrics: Epoch metric readings that get updated
        :param val_unsigned_images: Unsigned crops used to condition synthesis
        :param val_sub_targets: Ground truth substance labels
        :param image_out_val: Synthesized textures
        :param substance_out_val: Predicted substances
        :param extra: Extra info
        """
        image_gt_val = util.unsigned_to_signed(val_unsigned_images).to(self.device)

        # Preview deltas
        delta_c1_input, delta_c2_input, delta_c3_input = preview_deltas(extra["network_input"])
        delta_c1_output, delta_c2_output, delta_c3_output = preview_deltas(extra["network_output"])

        # Preview base color
        base_color_val = self._preview_base_color(val_unsigned_images, extra)

        # Style loss
        val_style_loss = utils_nt.get_loss(image_gt_val, image_out_val, self.train_params, self._vgg_features,
                                           self._gram_matrix, self._style_criterion)

        image_gt_val_for_preview = image_gt_val.clone()
        # Evaluate and preview substance
        if self.predictor.substances is not None:
            val_substance_loss = self._substance_criterion(substance_out_val,
                                                           val_sub_targets) * self.train_params.system.loss_params.substance_weight
            val_loss = val_style_loss + val_substance_loss
            epoch_metrics.epoch_substance_loss += val_substance_loss.item()

            _, val_substance_preds = substance_out_val.max(dim=1)
            val_substance_pass = (val_substance_preds == val_sub_targets).sum(dim=0)

            self._preview_substance_labels(val_substance_preds, image_gt_val_for_preview)

            epoch_metrics.epoch_substance_passed += val_substance_pass.item()
        else:
            val_loss = val_style_loss

        epoch_metrics.epoch_entry_count += image_out_val.shape[0]

        # Write all previews
        preview_val_imgs = preview_images(
            torch.cat([util.signed_to_unsigned(image_gt_val_for_preview).cpu(), base_color_val.cpu(),
                       delta_c1_input.cpu(), delta_c2_input.cpu(), delta_c3_input.cpu(),
                       util.signed_to_unsigned(image_out_val).cpu(),
                       delta_c1_output.cpu(), delta_c2_output.cpu(), delta_c3_output.cpu()], dim=0),
            self.train_params.train.bs)

        self.summary_writer.add_image("epoch_val_batch%d_results" % (val_i + 1), preview_val_imgs,
                                      global_step=epoch)

        # Update metrics
        epoch_metrics.epoch_loss += val_loss.item()
        epoch_metrics.epoch_style_loss += val_style_loss.item()
        epoch_metrics.epoch_batch_count += 1

    def _eval_epoch(self, epoch: int) -> TextureGenEpochSummary:
        """
        Evaluate model after an epoch
        :param epoch: Epoch id
        :return: Eval summary
        """
        epoch_metrics = TextureGenEpochSummary(epoch_loss=0.0, epoch_style_loss=0.0,
                                               epoch_substance_loss=0.0,
                                               epoch_batch_count=0, epoch_entry_count=0,
                                               epoch_substance_passed=0)

        with torch.no_grad():
            for val_i, val_batch in enumerate(self.val_dataloader):
                val_unsigned_images, val_filenames, val_sub_targets, val_unsigned_hsv_images = val_batch

                val_sub_targets = val_sub_targets.to(self.device)

                self.predictor.update_seed()

                val_sample_pos = self.predictor.get_position()

                predictor_result = self.predictor.predict(val_unsigned_images,
                                                          val_unsigned_hsv_images,
                                                          val_sample_pos,
                                                          train=False)

                self._update_eval_metrics(epoch, val_i, epoch_metrics, val_unsigned_images,
                                          val_sub_targets, predictor_result.image_out, predictor_result.substance_out, predictor_result.extra)

        return epoch_metrics

    def _get_checkpoint_metric_value(self, eval_stats: TextureGenEpochSummary) -> float:
        return eval_stats.epoch_style_loss

    def _report(self, epoch: int, train_stats: TextureGenEpochSummary, val_stats: TextureGenEpochSummary) -> None:
        """
        Report epoch train/val stats.
        :param epoch: Epoch index
        :param train_stats: Train summary
        :param val_stats: Val summary
        """
        self.summary_writer.add_scalar("epoch_training_loss", train_stats.epoch_loss / train_stats.epoch_batch_count,
                                       epoch)
        self.summary_writer.add_scalar("epoch_training_texture_loss",
                                       train_stats.epoch_style_loss / train_stats.epoch_batch_count,
                                       epoch)
        self.summary_writer.add_scalar("epoch_training_substance_loss",
                                       train_stats.epoch_substance_loss / train_stats.epoch_batch_count, epoch)

        if self.predictor.substances is not None:
            self.summary_writer.add_scalar("epoch_training_substance_acc",
                                           float(
                                               train_stats.epoch_substance_passed) / train_stats.epoch_entry_count,
                                           epoch)
            self.summary_writer.add_scalar("epoch_val_substance_acc",
                                           float(val_stats.epoch_substance_passed) / val_stats.epoch_entry_count,
                                           epoch)

        self.summary_writer.add_scalar("epoch_val_loss", val_stats.epoch_loss / val_stats.epoch_batch_count,
                                       epoch)
        self.summary_writer.add_scalar("epoch_val_texture_loss",
                                       val_stats.epoch_style_loss / val_stats.epoch_batch_count, epoch)
        self.summary_writer.add_scalar("epoch_val_substance_loss",
                                       val_stats.epoch_substance_loss / val_stats.epoch_batch_count,
                                       epoch)

        if self.predictor.substances is not None:
            logging.info(
                "[Epoch %d] train-loss: %.5f\t [sub: %.5f\t tex: %.5f]\t train-sub-acc: %.4f (%d/%d)\t "
                "val-loss: %.5f\t [sub: %.5f\t tex: %.5f]\t val-sub-ac: %.4f (%d/%d)" % (
                    epoch,
                    train_stats.epoch_loss / train_stats.epoch_batch_count,
                    train_stats.epoch_substance_loss / train_stats.epoch_batch_count,
                    train_stats.epoch_style_loss / train_stats.epoch_batch_count,
                    float(train_stats.epoch_substance_passed) / train_stats.epoch_entry_count,
                    train_stats.epoch_substance_passed,
                    train_stats.epoch_entry_count,
                    val_stats.epoch_loss / val_stats.epoch_batch_count,
                    val_stats.epoch_substance_loss / val_stats.epoch_batch_count,
                    val_stats.epoch_style_loss / val_stats.epoch_batch_count,
                    float(val_stats.epoch_substance_passed) / val_stats.epoch_entry_count,
                    val_stats.epoch_substance_passed,
                    val_stats.epoch_entry_count,
                ))
        else:
            logging.info(
                "[Epoch %d] train-loss: %.5f\t [tex: %.5f]\t val-loss: %.5f\t [tex: %.5f]" % (
                    epoch,
                    train_stats.epoch_loss / train_stats.epoch_batch_count,
                    train_stats.epoch_substance_loss / train_stats.epoch_batch_count,
                    val_stats.epoch_loss / val_stats.epoch_batch_count,
                    val_stats.epoch_style_loss / val_stats.epoch_batch_count,
                ))

    def _save_checkpoint(self, epoch: int, reason: SaveReason, train_stats, val_stats) -> None:
        """
        Saves a checkpoint
        :param epoch: Epoch
        :param reason: Save reason
        :param train_stats: Train stats
        :param val_stats: Val stats
        """
        if reason == SaveReason.BEST_MODEL:
            # Save best model
            logging.info("Best Style Loss")
            self.predictor.save_checkpoint(self.optimizer, epoch, self.epoch_stats[epoch],
                                           save_path=osp.join(self.output_path, "best_models",
                                                              "best-tex-val-loss-%.5f-epoch-%d.ckpt" % (
                                                                  val_stats.epoch_style_loss / val_stats.epoch_batch_count,
                                                                  epoch + 1)))
        elif reason == SaveReason.INTERVAL:
            logging.info("Saving Checkpoint")
            self.predictor.save_checkpoint(self.optimizer, epoch, self.epoch_stats[epoch],
                                           save_path=osp.join(self.output_path, "checkpoints",
                                                              "loss-%.5f-epoch-%d.ckpt" % (
                                                                  val_stats.epoch_style_loss / val_stats.epoch_batch_count,
                                                                  epoch)))

        else:
            assert False
