import logging

from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
from plan2scene.evaluation.matchers import AbstractMatcher
from plan2scene.utils.io import fraction_str
import numpy as np


class EvalResult:
    """
    Accumulates evaluation results.
    """

    def __init__(self, total_texture_loss: float, surface_count: int, house_room_surface_texture_loss_map: dict,
                 surface_total_texture_loss_map: dict, surface_count_map: dict):
        """
        Initialize eval results.
        :param total_texture_loss: Total texture loss
        :param surface_count: Number of surfaces contributed to loss
        :param house_room_surface_texture_loss_map: Mapping from house_key -> room_index -> surface -> loss
        :param surface_total_texture_loss_map: Mapping from surface -> loss. This accumulates loss from all surfaces of a particular type.
        :param surface_count_map: Mapping from surface -> surface count.
        """
        self.total_texture_loss = total_texture_loss
        self.surface_count = surface_count
        self.house_room_surface_texture_loss_map = house_room_surface_texture_loss_map
        self.surface_total_texture_loss_map = surface_total_texture_loss_map
        self.surface_count_map = surface_count_map
        self.entries = []

    def get_summary(self) -> dict:
        """
        Return summary dictionary
        :return: Summary dictionary
        """
        return {
            "total_texture_loss": self.total_texture_loss,
            "surface_count": self.surface_count,
            "house_room_surface_texture_loss_map": self.house_room_surface_texture_loss_map,
            "surface_total_texture_loss_map": self.surface_total_texture_loss_map,
            "surface_count_map": self.surface_count_map,
            "entries": self.entries
        }

    def __repr__(self):
        # Verify Average
        losses = [a["loss"] for a in self.entries]
        assert np.abs(np.sum(losses).item() - self.total_texture_loss) < 0.00001
        assert len(losses) == self.surface_count

        return "%s\t Floor: %s\t Wall: %s\t Ceiling: %s" % (
            fraction_str(self.total_texture_loss, self.surface_count),
            fraction_str(self.surface_total_texture_loss_map["floor"], self.surface_count_map["floor"]),
            fraction_str(self.surface_total_texture_loss_map["wall"], self.surface_count_map["wall"]),
            fraction_str(self.surface_total_texture_loss_map["ceiling"], self.surface_count_map["ceiling"]),
        )

    def get_house_results(self, house_key: str):
        """
        Separate results specific to a house.
        :param house_key: House key of the interested house.
        :return: Results of the specified house.
        """
        eval_results = EvalResult.new()
        if house_key not in self.house_room_surface_texture_loss_map:
            return None
        house_results = self.house_room_surface_texture_loss_map[house_key]
        eval_results.house_room_surface_texture_loss_map = {house_key: house_results}

        for room_index, room in house_results.items():
            for surface, loss in room.items():
                eval_results.total_texture_loss += loss
                eval_results.surface_count += 1
                eval_results.surface_count_map[surface] += 1
                eval_results.surface_total_texture_loss_map[surface] += loss
        eval_results.entries = [a for a in self.entries if a["house_key"] == house_key]
        return eval_results

    @classmethod
    def new(cls):
        """
        Initializes a new EvalResults with empty data.
        :return: EvalResults
        """
        return EvalResult(total_texture_loss=0, surface_count=0, house_room_surface_texture_loss_map={},
                          surface_total_texture_loss_map={"floor": 0, "wall": 0, "ceiling": 0},
                          surface_count_map={"floor": 0, "wall": 0, "ceiling": 0})

    def append_results(self, texture_loss, surface: str, house_key: str, room_index: int) -> None:
        """
        Append metrics of a room surface.
        :param texture_loss: Loss computed for the surface
        :param surface: Considered surface type.
        :param house_key: House key of considered house.
        :param room_index: Index of considered room.
        """
        self.total_texture_loss += texture_loss
        self.surface_count += 1
        self.surface_count_map[surface] += 1
        self.surface_total_texture_loss_map[surface] += texture_loss

        if house_key not in self.house_room_surface_texture_loss_map:
            self.house_room_surface_texture_loss_map[house_key] = {}
        room_surface_texture_loss_map = self.house_room_surface_texture_loss_map[house_key]

        if room_index not in room_surface_texture_loss_map:
            room_surface_texture_loss_map[room_index] = {}

        surface_texture_loss_map = room_surface_texture_loss_map[room_index]

        assert surface not in surface_texture_loss_map
        surface_texture_loss_map[surface] = texture_loss
        self.entries.append({
            "loss": texture_loss,
            "surface": surface,
            "house_key": house_key,
            "room_index": room_index
        })


def evaluate_surface(house_key, room_index, surface_name: str, pred_textures: map, gt_textures: map,
                     matcher: AbstractMatcher, eval_results: EvalResult = None,
                     key: str = "prop"):
    """
    Evaluates texture prediction of a surface. Update eval results (if specified). Return eval results.
    :param house_key: Key of the concerned house.
    :param room_index: Index of the concerned room.
    :param surface_name: Surface name.
    :param pred_textures: Predicted textures for the surface.
    :param gt_textures: Ground truth textures for the surface.
    :param matcher: Matcher used to match ground prediction with ground truth.
    :param eval_results: Optional. Eval results to update.
    :param key: Key used to identify the predictions.
    :return: Updated eval results.
    """
    if eval_results is None:
        eval_results = EvalResult.new()

    if key in pred_textures:
        pred = pred_textures[key]
        texture_loss, gt_match, matched = matcher(pred, gt_textures)
        if matched:
            eval_results.append_results(texture_loss=texture_loss, surface=surface_name, house_key=house_key,
                                        room_index=room_index)

    return eval_results


def eval_room(conf: ConfigManager, house_key: str, pred_room: Room, gt_room: Room, matcher: AbstractMatcher,
              eval_results: EvalResult = None, key: str = "prop"):
    """
    Evaluate texture predictions of a given room. Update eval results (if specified). Return eval results.
    :param conf: Config manager.
    :param house_key: Key of the house.
    :param pred_room: Room with predicted textures.
    :param gt_room: Room with ground truth references.
    :param matcher: Matcher used to match ground truth with predictions.
    :param eval_results: Optional. Eval results to update.
    :param key: Key used to identify predictions.
    :return: Updated eval results
    """
    if eval_results is None:
        eval_results = EvalResult.new()

    assert isinstance(pred_room, Room)
    assert isinstance(gt_room, Room)

    assert pred_room.room_index == gt_room.room_index

    for surf in conf.surfaces:
        evaluate_surface(house_key, pred_room.room_index, surf, pred_room.surface_textures[surf],
                         gt_room.surface_textures[surf],
                         matcher, eval_results, key)
    return eval_results


def eval_house(conf: ConfigManager, pred_house: House, gt_house: House, matcher: AbstractMatcher,
               eval_results: EvalResult = None, key: str = "prop"):
    """
    Evaluate a house. Update eval results (if specified). Return eval results.
    :param conf: Config manager
    :param pred_house: House with predicted textures.
    :param gt_house: House with ground truth reference crops.
    :param matcher: Matcher used to match ground truth with predictions.
    :param eval_results: Optional. Eval results to update.
    :param key: Key used to identify predictions.
    :return: Updated eval results.
    """
    if eval_results is None:
        eval_results = EvalResult.new()

    assert pred_house.house_key == gt_house.house_key

    for room_index, pred_room in pred_house.rooms.items():
        gt_room = gt_house.rooms[room_index]

        eval_room(conf=conf, house_key=pred_house.house_key, pred_room=pred_room, gt_room=gt_room, matcher=matcher,
                  eval_results=eval_results, key=key)

    return eval_results


def evaluate(conf: ConfigManager, pred_houses: dict, gt_houses: dict, matcher: AbstractMatcher, key: str = "prop",
             log: bool = True) -> EvalResult:
    """
    Evaluate a list of houses.
    :param conf: Config manager
    :param pred_houses: Dictionary of houses with predicted textures.
    :param gt_houses: Dictionary of houses with ground truth reference.
    :param matcher: Matcher used to match ground truth with predictions.
    :param key: Key used to identify predictions.
    :param log: Specify true to log progress.
    :return: Eval results.
    """
    eval_results = EvalResult.new()

    for i, (house_key, pred_house) in enumerate(pred_houses.items()):
        if log:
            logging.info("[%d/%d] Evaluating %s" % (i, len(pred_houses), house_key))
        gt_house = gt_houses[house_key]
        eval_house(conf=conf, pred_house=pred_house, gt_house=gt_house, matcher=matcher, key=key,
                   eval_results=eval_results)

    return eval_results
