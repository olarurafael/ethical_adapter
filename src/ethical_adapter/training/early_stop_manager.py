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

        self.stop_file = self.run_dir / "STOP"
        if self.stop_file.exists():
            self.stop_file.unlink(missing_ok=True)

        # Hook Ctrl+C
        signal.signal(signal.SIGINT, self._on_sigint)

    # ------------------------------
    # manual stop
    # ------------------------------
    def _on_sigint(self, signum, frame):
        self.manual_flag = True
        print("\n[INFO] Early stop requested — will save after this epoch.")

    def manual_stop_requested(self):
        """Return True if STOP file exists or Ctrl+C pressed."""
        return self.manual_flag or self.stop_file.exists()

    # ------------------------------
    # metric stop
    # ------------------------------
    def update(self, val_loss, epoch):
        """Update metric-based logic; return True if stop triggered."""
        if not self.enabled:
            return False

        if val_loss < (self.best_val - self.min_delta):
            self.best_val = val_loss
            self.best_epoch = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1

        return self.no_improve >= self.patience

