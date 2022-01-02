import logging
logger = logging.getLogger('my_logger')


class EarlyStopping:
    def __init__(self, mode, patience=10):
        self.patience = patience
        self.mode = mode
        self.min, self.max = float('inf'), 0
        self.epochs_no_improve = 0

    @property
    def should_stop(self):
        if self.epochs_no_improve == self.patience:
            logging.info('Early stopping!')
            return True
        return False

    def has_improved(self, val):
        if self._is_better_result(val):
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        logging.info(f"epochs_no_improve: {self.epochs_no_improve}/{self.patience}\n")
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
