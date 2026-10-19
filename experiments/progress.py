"""
progress.py
===========
Tiny shared progress-bar helper used by training and evaluation, so both show a
consistent live bar with percentage, counts, rate, and ETA.
"""

import sys
import time


class ProgressBar:
    """
    Single-line, self-overwriting progress bar.

      pb = ProgressBar(total, prefix="train")
      for ... :
          ...
          pb.update(extra=f"loss={loss:.4f}")
      pb.close()
    """
    def __init__(self, total, prefix="", width=32, stream=sys.stdout):
        self.total  = max(int(total), 1)
        self.prefix = prefix
        self.width  = width
        self.stream = stream
        self.n      = 0
        self.t0     = time.time()

    def update(self, step=1, extra=""):
        self.n += step
        self._draw(extra)

    def set(self, n, extra=""):
        self.n = n
        self._draw(extra)

    def _draw(self, extra=""):
        frac    = min(self.n / self.total, 1.0)
        filled  = int(self.width * frac)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.t0
        rate    = self.n / max(elapsed, 1e-6)
        eta     = (self.total - self.n) / max(rate, 1e-6)
        msg = (f"\r  {self.prefix} [{bar}] {frac*100:5.1f}% "
               f"{self.n}/{self.total}  {rate:5.1f} it/s  "
               f"ETA {eta:4.0f}s")
        if extra:
            msg += f"  {extra}"
        self.stream.write(msg)
        self.stream.flush()

    def close(self, extra=""):
        self._draw(extra)
        self.stream.write("\n")
        self.stream.flush()
