import enum


class SaveReason(enum.Enum):
    """
    Specify the reason to save a checkpoint
    """
    BEST_MODEL = "best_model"
    INTERVAL = "interval"
