import numpy as np

from .metrics import MetricsMeter


class EarlyStopping:
    def __init__(self, logger, mode, patience=10):
        self.logger = logger
        self.patience = patience
        self.mode = mode
        self.min, self.max = float('inf'), 0
        self.epochs_no_improve = 0

    @property
    def should_stop(self):
        if self.epochs_no_improve == self.patience:
            self.logger.info('Early stopping!')
            return True
        return False

    def has_improved(self, val):
        if self._is_better_result(val):
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        self.logger.info(f"epochs_no_improve: {self.epochs_no_improve}/{self.patience}\n")
        return False

    def _is_better_result(self, val):
        if self.mode == "min":
            if val < self.min:
                self.min = val
                return True
        else:
            if val > self.max:
                self.max = val
                return True
        return False


def update_dict(metrics: MetricsMeter, d: dict):
    d["accuracy"].append(metrics.accuracy)
    d["precision"].append(metrics.precision)
    d["recall"].append(metrics.recall)
    d["F1"].append(metrics.f1)
    d["AUC"].append(metrics.auc)


def mean_over_run(logger, d: dict):
    res = ""
    runs = -1
    for k, v in d.items():
        tmp = np.array(d[k])
        runs = tmp.shape[0]
        res += f"mean: {tmp.mean():.3f}, std: {tmp.std():.3f} - {k} {tmp}\n"
    logger.info(f"mean and std over {runs} runs")
    logger.info(res)
