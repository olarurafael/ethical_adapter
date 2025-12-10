# src/ethical_adapter/early_stop_manager.py
import signal
from pathlib import Path


class EarlyStopManager:
    """
    Unified early-stopping helper:
      - supports Ctrl+C (SIGINT) or touch STOP file for manual stop
      - supports patience-based metric early stop
    """

    def __init__(self, run_dir, enabled=False, patience=1, min_delta=0.0):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta

        self.best_val = float("inf")
        self.best_epoch = -1
        self.no_improve = 0

        self.manual_flag = False
        self.reason = None  # "metric" or "manual" or "stop_file"

        self.stop_file = self.run_dir / "STOP"
        if self.stop_file.exists():
            self.stop_file.unlink(missing_ok=True)

        # Hook Ctrl+C
        signal.signal(signal.SIGINT, self._on_sigint)

    # manual stop requested
    def _on_sigint(self, signum, frame):
        self.manual_flag = True
        print("\n Early stop requested — will save after this epoch.")

    def _check_manual(self):
        if self.manual_flag:
            self.reason = "manual"
            return True

        if self.stop_file.exists():
            self.reason = "stop_file"
            return True

        return False

    def _check_metric(self, val_loss, epoch):
        if not self.enabled:
            return False

        improved = val_loss < (self.best_val - self.min_delta)

        if improved:
            self.best_val = val_loss
            self.best_epoch = epoch
            self.no_improve = 0
            return False  # don't stop yet
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                self.reason = "metric"
                return True

        return False

    # unified check
    def should_stop(self, val_loss=None, epoch=None):
        """
        returns true if stopping is needed for any reason.
        self.reason is gives the type of stop
        """

        if self._check_manual():
            return True

        if val_loss is not None:
            if self._check_metric(val_loss, epoch):
                return True

        return False
