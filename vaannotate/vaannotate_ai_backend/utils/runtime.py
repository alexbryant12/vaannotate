"""Runtime utilities for logging, cancellation, and progress reporting."""
from __future__ import annotations

import json
import logging
import sys as _sys
import time
from contextlib import contextmanager
from typing import Callable, Optional

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        for k, v in getattr(record, "__dict__", {}).items():
            if k in (
                "args",
                "msg",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                continue
            if k.startswith("_"):
                continue
            if k in base:
                continue
            try:
                json.dumps({k: v})
                base[k] = v
            except Exception:
                base[k] = str(v)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    return logger


class CancelledError(RuntimeError):
    """Raised when a cancellation request is received."""


_cancel_check: Optional[Callable[[], bool]] = None


@contextmanager
def cancellation_scope(callback: Optional[Callable[[], bool]]):
    """Temporarily install a cancellation callback for long-running loops."""

    global _cancel_check
    previous = _cancel_check
    _cancel_check = callback
    try:
        yield
    finally:
        _cancel_check = previous


def check_cancelled() -> None:
    if _cancel_check and _cancel_check():
        raise CancelledError("AI backend run cancelled")


LOGGER = setup_logging()


# ---- Pretty progress logging (ETA) -------------------------------------------

def _fmt_hms(_secs: float) -> str:
    if not _secs or _secs == float("inf") or _secs != _secs:  # NaN
        return "--:--:--"
    secs = int(max(0, _secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar_str(fraction: float, width: int = 32, ascii_only: bool = False) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    if ascii_only:
        mid = ">" if 0 < filled < width else ""
        return "[" + "#" * max(0, filled - 1) + mid + "." * (width - filled) + "]"
    # Unicode blocks
    full = "█"
    empty = "·"
    return "[" + full * filled + empty * (width - filled) + "]"


def iter_with_bar(
    step: str,
    iterable,
    *,
    total: int | None = None,
    bar_width: int = 32,
    min_interval_s: float = 10,
    ascii_only: bool | None = None,
):
    """
    Wrap an iterable and render a live progress bar if stderr is a TTY.
    Falls back to periodic log lines otherwise.
    """

    try:
        if total is None:
            total = len(iterable)  # may fail for generators
    except Exception:
        total = None

    t0 = time.time()
    last = t0
    tty = hasattr(_sys.stderr, "isatty") and _sys.stderr.isatty()
    if ascii_only is None:
        ascii_only = not tty  # default: Unicode in TTY

    wrote_progress = False
    last_render_len = 0
    last_tty_msg = ""
    for i, item in enumerate(iterable, 1):
        check_cancelled()
        now = time.time()
        if tty and (i == 1 or now - last >= min_interval_s or (total and i == total)):
            last = now
            elapsed = now - t0
            rate = (i / elapsed) if elapsed > 0 else 0.0
            if total:
                frac = i / total
                eta = ((total - i) / rate) if rate > 0 else float("inf")
                bar = _bar_str(frac, width=bar_width, ascii_only=ascii_only)
                msg = (
                    f"{step:<14} {bar}  {int(frac*100):3d}%  {i}/{total} • {rate:.2f}/s • ETA {_fmt_hms(eta)}"
                )
            else:
                spinner = "-\\|/"[int((now - t0) * 8) % 4]
                msg = f"{step:<14} [{spinner}]  {i} done • {rate:.2f}/s • elapsed {_fmt_hms(now - t0)}"
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            _sys.stderr.write("\r" + msg)
            last_render_len = len(msg)
            last_tty_msg = msg
            wrote_progress = True
            _sys.stderr.flush()
        elif not tty and (i == 1 or now - last >= min_interval_s or (total and i == total)):
            last = now
            elapsed = now - t0
            rate = (i / elapsed) if elapsed > 0 else 0.0
            if total:
                eta = ((total - i) / rate) if rate > 0 else float("inf")
                msg = f"[{step}] {i}/{total} • {rate:.2f}/s • ETA {_fmt_hms(eta)}"
            else:
                msg = f"[{step}] {i} done • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}"
            _sys.stderr.write(msg + "\n")
            _sys.stderr.flush()
        yield item

    # finish line
    if tty:
        if total:
            elapsed = time.time() - t0
            rate = (total / elapsed) if elapsed > 0 else 0.0
            bar = _bar_str(1.0, width=bar_width, ascii_only=ascii_only)
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            final = f"{step:<14} {bar}  100%  {total}/{total} • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}"
            _sys.stderr.write("\r" + final + "\n")
            last_tty_msg = final
        else:
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            if wrote_progress and last_tty_msg:
                _sys.stderr.write("\r" + last_tty_msg)
            _sys.stderr.write("\n")
        _sys.stderr.flush()
