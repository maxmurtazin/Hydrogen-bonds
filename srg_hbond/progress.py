from __future__ import annotations
import sys, time
from dataclasses import dataclass, field
from collections import deque


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f"{seconds:5.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m:02d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


@dataclass
class ETA:
    total: int
    window: int = 30
    start: float = field(default_factory=time.time)
    times: deque = field(default_factory=deque)
    last: float = field(default_factory=time.time)

    def tick(self, step: int) -> dict:
        now = time.time()
        dt = now - self.last
        self.last = now
        self.times.append(dt)
        while len(self.times) > self.window:
            self.times.popleft()
        done = step + 1
        elapsed = now - self.start
        avg_recent = sum(self.times) / max(1, len(self.times))
        avg_global = elapsed / max(1, done)
        # Blend recent and global to reduce jitter while reacting to slowdowns.
        avg = 0.65 * avg_recent + 0.35 * avg_global
        remaining = max(0, self.total - done)
        eta = remaining * avg
        progress = done / max(1, self.total)
        return {
            "step_time_s": dt,
            "avg_step_time_s": avg,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "progress": progress,
            "steps_per_s": 1.0 / avg if avg > 1e-12 else 0.0,
        }


def print_progress(step: int, total: int, metrics: dict, prefix: str = "") -> None:
    width = 28
    p = metrics.get("progress", (step + 1) / max(1, total))
    filled = int(width * p)
    bar = "█" * filled + "░" * (width - filled)
    msg = (
        f"\r{prefix}[{step+1:>5}/{total:<5}] {bar} "
        f"{100*p:6.2f}%  ETA={fmt_seconds(metrics.get('eta_s', 0))}  "
        f"elapsed={fmt_seconds(metrics.get('elapsed_s', 0))}  "
        f"{metrics.get('steps_per_s', 0):6.2f} step/s"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()
    if step + 1 >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()
