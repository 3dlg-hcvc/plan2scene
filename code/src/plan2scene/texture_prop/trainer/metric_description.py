from plan2scene.evaluation.evaluator import EvalResult
from plan2scene.evaluation.matchers import AbstractMatcher


class MetricDescription:
    """
    Describes a metric using its name and the evaluator.
    """
    def __init__(self, name: str, evaluator: AbstractMatcher):
        """
        Initialize metric description.
        :param name: Metric name.
        :param evaluator: Matcher used to evaluate the metric.
        """
        self._name = name
        self._evaluator = evaluator

    @property
    def name(self) -> str:
        """
        Return metric name.
        :return: Metric name.
        """
        return self._name

    @property
    def evaluator(self) -> AbstractMatcher:
        """
        Return metric evaluator.
        :return: Metric evaluator.
        """
        return self._evaluator

    def __repr__(self):
        return self.name


class MetricResult:
    """
    Contains evaluation results for a metric.  Pairs a metric description with the evaluation result.
    """
    def __init__(self, metric: MetricDescription, eval_result: EvalResult):
        """
        Initialize metric result.
        :param metric: Metric considered.
        :param eval_result: Evaluation result reported by the metric.
        """
        self._metric = metric
        self._eval_result = eval_result

    @property
    def metric(self) -> MetricDescription:
        """
        Return metric description.
        :return: Metric description.
        """
        return self._metric

    @property
    def eval_result(self) -> EvalResult:
        """
        Return evaluation result.
        :return: Evaluation result.
        """
        return self._eval_result
