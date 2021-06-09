import os.path as osp
import os
import logging

from plan2scene.common.color_description import ColorSpace, Color
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.nets.neural_texture.texture_gen import TextureGen
from plan2scene.texture_gen.predictor_result import TextureGenPredictorResult
from plan2scene.texture_gen.utils.hsv_utils import hsv_decompose_median, hsv_recombine_median, hsv_to_rgb, \
    rgb_decompose_median, rgb_recombine_median, rgb_to_hsv

from torch import nn, Tensor
from torch import optim
import plan2scene.texture_gen.utils.neural_texture_helper as utils_nt
from plan2scene.texture_gen.utils.neural_texture_helper import get_loss_no_reduce
import torch
import plan2scene.texture_gen.utils.utils as util

from plan2scene.texture_gen.custom_transforms.hsv_transforms import ToHSV
from torchvision import transforms as tfs


class TextureGenPredictor:
    """
    Predicts textures using our modified neural texture synthesis approach.
    """

    def __init__(self, conf, rgb_median_emb: bool):
        """
        Initializes predictor
        :param conf: Model config used by neural texture synthesis.
        :param rgb_median_emb: Should the color component of the median embedding be converted back to RGB?
        """
        self.conf = conf
        self.rgb_median_emb = rgb_median_emb

        # Modified neural texture network
        self.net = TextureGen(conf).to(conf.device)

        # Load substance labels if supported
        self.substances = None
        if "substances" in conf.dataset and conf.system.arch.model_substance_classifier.model_params.available:
            self.substances = conf.dataset.substances

        self.update_seed()

        self.vgg_features = utils_nt.VGGFeatures().to(conf.device)
        self.gram_matrix = utils_nt.GramMatrix().to(conf.device)
        self.criterion = nn.MSELoss(reduction="none")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load saved checkpoint
        :param checkpoint_path: Path to checkpoint
        """
        ckpt = torch.load(checkpoint_path)
        logging.info(
            "Loading Checkpoint %s: %s" % (checkpoint_path, self.net.load_state_dict(ckpt["model_state_dict"])))

    def get_position(self) -> Tensor:
        """
        Return the position tensor
        :return: position tensor
        """
        sample_pos = utils_nt.get_position((self.conf.image.image_res, self.conf.image.image_res), self.conf.dim,
                                           self.conf.device, self.conf.train.bs)
        return sample_pos

    def predict_embs(self, sample_image_crops: list):
        """
        Predict latent embeddings and VGG losses for the given crops using the encoder.
        :param sample_image_crops: List of PIL Image crops
        :return: Tuple (embeddings tensor, losses tensor)
        """
        with torch.no_grad():
            unsigned_images = torch.cat([tfs.ToTensor()(a).unsqueeze(0) for a in sample_image_crops], dim=0)
            unsigned_hsv_images = torch.cat([tfs.ToTensor()(ToHSV()(a)).unsqueeze(0) for a in sample_image_crops],
                                            dim=0)

            # Predict using the network
            predictor_result = self.predict(unsigned_images.to(self.conf.device),
                                            unsigned_hsv_images.to(self.conf.device),
                                            self.get_position(), combined_emb=None, train=False)

            # Compute loss between synthesized texture and conditioned image
            losses = get_loss_no_reduce(util.unsigned_to_signed(unsigned_images).to(self.conf.device), predictor_result.image_out,
                                        self.conf, self.vgg_features, self.gram_matrix, self.criterion)
            return predictor_result.combined_emb.cpu(), losses.cpu()

    def predict_textures(self, combined_embs: Tensor, multiplier: int) -> tuple:
        """
        Synthesize a texture given the latent embedding.
        :param combined_embs: Latent embedding used to condition texture synthesis
        :param multiplier: Multiplier on output size. We make multiple predictions and stitch. (E.g. if multiplier is 2, we stitch a 2x2 texture)
        :return: Tuple of (Predicted textures as a list of PIL images, List of predicted substance labels, extra)
        """
        assert isinstance(multiplier, int)
        with torch.no_grad():
            if isinstance(combined_embs, list):
                combined_embs = torch.cat([a for a in combined_embs], dim=0)

            # Identify substance prediction
            predictor_result = self.predict(unsigned_images=None, unsigned_hsv_images=None,
                                            sample_pos=self.get_position(), train=False,
                                            combined_emb=combined_embs)

            substance_names = None
            if predictor_result.substance_out is not None:
                substance_out = predictor_result.substance_out.cpu()
                _, substance_preds = substance_out.max(dim=1)
                substance_names = [self.substances[a.item()] for a in substance_preds]
            else:
                # No substance prediction
                substance_names = [None for a in range(combined_embs.shape[0])]

            # Generate a large texture by stitching multiple texture predictions
            base_position_stripe = self.get_position()
            y_stripe = []
            # Loop for stitching along y axis
            for y in range(multiplier):
                y_position_stripe = torch.cat(
                    [base_position_stripe[:, 0:1], base_position_stripe[:, 1:2] - 2.0 * y, base_position_stripe[:, 2:]],
                    dim=1)
                x_stripe = []
                for x in range(multiplier):
                    # Loop for stitching along x axis
                    position_stripe = torch.cat(
                        [y_position_stripe[:, 0:0], y_position_stripe[:, 0:1] + 2.0 * x, y_position_stripe[:, 1:]],
                        dim=1)

                    # Prediction of a texture crop
                    texture_pred_results = self.predict(unsigned_images=None, unsigned_hsv_images=None,
                                                        sample_pos=position_stripe, train=False,
                                                        combined_emb=combined_embs)
                    x_stripe.append(texture_pred_results.image_out.cpu())

                x_stripe = torch.cat(x_stripe, dim=3)  # Merge along x axis
                y_stripe.append(x_stripe)

            y_stripe = torch.cat(y_stripe, dim=2)  # Merge along y axis
            y_stripe = util.signed_to_unsigned(y_stripe)
            y_stripe = [tfs.ToPILImage()(a) for a in y_stripe]  # Convert to PIL images

            return y_stripe, substance_names, predictor_result.extra

    def _compute_network_input(self, unsigned_images: Tensor, unsigned_hsv_images: Tensor, additional_params) -> tuple:
        """
        Compute network input and base color
        :param unsigned_images: Tensor of unsigned RGB images
        :param unsigned_hsv_images: Tensor of unsigned HSV images
        :param additional_params: Additional params to merge to latent embedding. This method may update this.
        :return: Tuple (network input, base_color)
        """
        assert int(self.conf.image.hsv_decomp) + int(self.conf.image.hsv) + int(self.conf.image.rgb_decomp) <= 1

        if self.conf.image.hsv_decomp or self.conf.image.hsv:
            # HSV_DECOMP: Convert input to HSV and separate median color. (Paper method)
            # HSV: Convert input to HSV. (Ablation reported in paper)

            image_gt_decomposed, median_h, median_s, median_v = hsv_decompose_median(
                unsigned_hsv_images.to(self.conf.device))
            base_color = Color(color_space=ColorSpace.HSV, components=[median_h, median_s, median_v])

            if self.conf.image.hsv and not self.conf.image.hsv_decomp:
                network_input = util.unsigned_to_signed(unsigned_hsv_images.to(self.conf.device))
            else:
                if self.rgb_median_emb:  # Should the color info of combined emb be in RGB?
                    rgb_median = hsv_to_rgb(
                        torch.cat([median_h.unsqueeze(1), median_s.unsqueeze(1), median_v.unsqueeze(1)],
                                  dim=1).unsqueeze(2).unsqueeze(3))
                    additional_params.extend(
                        [rgb_median[:, 0, 0, 0], rgb_median[:, 1, 0, 0], rgb_median[:, 2, 0, 0]])
                else:
                    additional_params.extend([median_h, median_s, median_v])
                network_input = image_gt_decomposed
        elif self.conf.image.rgb_decomp:
            # RGB_DECOMP: Keep network input in RGB. But separate median color.

            image_gt_decomposed, median_r, median_g, median_b = rgb_decompose_median(
                unsigned_images.to(self.conf.device))
            additional_params.extend([median_r, median_g, median_b])
            base_color = Color(color_space=ColorSpace.RGB, components=[median_r, median_g, median_b])
            network_input = image_gt_decomposed
        else:
            # Use RGB tensor as the network input
            base_color = None
            network_input = util.unsigned_to_signed(unsigned_images).to(self.conf.device)
        return network_input, base_color

    def _compute_network_emb(self, combined_emb: Tensor, additional_params: list) -> tuple:
        """
        Compute network embedding given the combined embedding
        :param combined_emb: Combined embedding including the network embedding and base color
        :return: tuple (Network embedding, base_colour)
        """
        if self.conf.image.hsv_decomp:
            if self.rgb_median_emb:
                hsv_median = rgb_to_hsv(combined_emb[:, -3:].unsqueeze(2).unsqueeze(3))
                median_h = hsv_median[:, 0, 0, 0]
                median_s = hsv_median[:, 1, 0, 0]
                median_v = hsv_median[:, 2, 0, 0]
                additional_params.extend([combined_emb[:, -3], combined_emb[:, -2], combined_emb[:, -1]])
            else:
                median_h = combined_emb[:, -3]
                median_s = combined_emb[:, -2]
                median_v = combined_emb[:, -1]
                additional_params.extend([median_h, median_s, median_v])
            embedding = combined_emb[:, :-3]
            base_color = Color(color_space=ColorSpace.HSV, components=[median_h, median_s, median_v])
        elif self.conf.image.hsv:
            embedding = combined_emb[:, :]
            base_color = None
        elif self.conf.image.rgb_decomp:
            median_r = combined_emb[:, -3]
            median_g = combined_emb[:, -2]
            median_b = combined_emb[:, -1]
            additional_params.extend([median_r, median_g, median_b])
            embedding = combined_emb[:, :-3]
            base_color = Color(color_space=ColorSpace.RGB, components=[median_r, median_g, median_b])
        else:
            # RGB
            embedding = combined_emb[:, :]
            base_color = None
        return embedding, base_color

    def _compute_image_from_net_output(self, network_out: Tensor, base_color: Color) -> Tensor:
        """
        Compute output image given the network output and the base colour
        :param network_out: Output from the network
        :param base_colour: Base colour
        :return: Output image as a signed tensor
        """
        if self.conf.image.hsv_decomp or self.conf.image.hsv:
            assert base_color.color_space == ColorSpace.HSV
            median_h, median_s, median_v = base_color.components
            if self.conf.image.hsv and not self.conf.image.hsv_decomp:
                image_out = util.signed_to_unsigned(network_out)
            else:
                image_out = hsv_recombine_median(network_out, median_h, median_s, median_v)
            image_out = hsv_to_rgb(image_out)  # Differentiable HSV to RGB layer
            image_out = util.unsigned_to_signed(image_out)
        elif self.conf.image.rgb_decomp:
            assert base_color.color_space == ColorSpace.RGB
            median_r, median_g, median_b = base_color.components
            image_out = rgb_recombine_median(network_out, median_r, median_g, median_b)
            image_out = util.unsigned_to_signed(image_out)
        else:
            image_out = network_out
        return image_out

    def predict(self, unsigned_images: Tensor, unsigned_hsv_images: Tensor, sample_pos: Tensor, train: bool,
                combined_emb: Tensor = None) -> TextureGenPredictorResult:
        """
        Predict textures and embeddings given input crops.
        :param unsigned_images: Input crops in unsigned RGB
        :param unsigned_hsv_images: Input crops in unsigned HSV
        :param sample_pos: Position tensor
        :param train: Is train mode?
        :param combined_emb: If provided, we skip the encoder and directly pass the embedding to the decoder.
        :return: Predictor results
        """
        if train:
            self.net.train()
        else:
            self.net.eval()

        # Additional params gets concatenated to the latent embedding given by the encoder
        additional_params = []

        if combined_emb is None:
            # Predict using the image input. Use encoder.
            network_input, base_color = self._compute_network_input(unsigned_images, unsigned_hsv_images, additional_params)
            network_out, network_emb, substance_out = self.net(network_input, sample_pos.to(self.conf.device),
                                                               self.seed)
        else:
            # Predict using the combined_emb. Skip encoder.
            network_input = None
            combined_emb = combined_emb.to(self.conf.device)
            network_emb, base_color = self._compute_network_emb(combined_emb, additional_params)

            network_out, network_emb, substance_out = self.net(None, sample_pos.to(self.conf.device), self.seed,
                                                               weights_bottleneck=network_emb)

        # Compute network output
        image_out = self._compute_image_from_net_output(network_out, base_color)

        extra = {
            "network_input": network_input,
            "network_output": network_out,
            "network_emb": network_emb,
            "base_color": base_color
        }

        # Compute combined_emb
        combined_emb = network_emb
        if len(additional_params) > 0:
            additional_params_emb = torch.cat([a.unsqueeze(1) for a in additional_params], dim=1)
            combined_emb = torch.cat([combined_emb, additional_params_emb], dim=1)
        return TextureGenPredictorResult(image_out=image_out, combined_emb=combined_emb, substance_out=substance_out, extra=extra)

    def update_seed(self) -> None:
        """
        Update seed used to synthesize textures
        """
        self.seed = torch.rand((self.conf.train.bs, self.conf.noise.octaves, self.conf.texture.channels),
                               device=self.conf.device)

    def save_checkpoint(self, opt: optim.Adam, epoch: int, stats: dict, save_path) -> None:
        """
        Save model checkpoint
        :param opt: optimizer
        :param epoch: epoch
        :param stats: Stats dictionary
        :param save_path: Save path
        """
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        payload = {
            "model_state_dict": self.net.state_dict(),
            "epoch": epoch,
            "stats": stats,
            "optimizer_state_dict": opt.state_dict()
        }
        torch.save(payload, save_path)
