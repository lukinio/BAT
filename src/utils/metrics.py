import torch
from sklearn.metrics import roc_auc_score


class MetricsMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total, self.total_loss = 0., 0.
        self.tp, self.fn = 0., 0.
        self.fp, self.tn = 0., 0.
        self.y_true, self.y_pred = [], []

    def _update_loss(self, val, n=1):
        self.total_loss += val * n

    def _update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred += y_pred.cpu().tolist()
        self.y_true += y_true.cpu().tolist()

        self.tp += (y_true * y_pred).sum().item()
        self.tn += ((1 - y_true) * (1 - y_pred)).sum().item()
        self.fp += ((1 - y_true) * y_pred).sum().item()
        self.fn += (y_true * (1 - y_pred)).sum().item()

    def update(self, y_pred, y_true, val):
        n = y_true.size(0)
        self.total += n
        self._update_metrics(y_pred, y_true)
        self._update_loss(val, n)

    @property
    def show(self):
        return f"Loss: {self.loss:.4f}, accuracy: {self.accuracy:.3f}, F1: {self.f1:.3f}, AUC: {self.auc:.3f}"

    @property
    def loss(self):
        return self.total_loss / self.total

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.total * 100

    @property
    def precision(self):
        try:
            return self.tp / (self.tp + self.fp) * 100
        except ZeroDivisionError:
            return 0

    @property
    def recall(self):
        try:
            return self.tp / (self.tp + self.fn) * 100
        except ZeroDivisionError:
            return 0

    @property
    def f1(self):
        try:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        except ZeroDivisionError:
            return 0

    @property
    def auc(self):
        return roc_auc_score(self.y_true, self.y_pred)
