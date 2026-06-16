"""v17 logging — deliberately verbose. Every module logs through here so an
iteration's full reasoning is reconstructable from the .log file alone.

Design goals (per the v17 brief: "excessive logging in every code"):
  * one logger per (iteration, component), tee'd to console + per-iter file
  * millisecond timestamps + component tag + level on every line
  * counters: lightweight named counters you can bump in hot loops and dump
  * timed(): context manager that logs enter/exit + wall time
  * every log line is greppable: tags are UPPERCASE in [brackets]
"""
from __future__ import annotations
import logging, os, sys, time, json
from contextlib import contextmanager
from collections import defaultdict

_LOG_DIR = os.environ.get("V17_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(_LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)-18s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str, iter_tag: str | None = None, console: bool = True) -> logging.Logger:
    """Return a logger that writes to logs/<iter_tag>.log (+ console)."""
    full = name if iter_tag is None else f"{iter_tag}:{name}"
    lg = logging.getLogger(full)
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)
    fname = os.path.join(_LOG_DIR, f"{iter_tag or 'v17'}.log")
    fh = logging.FileHandler(fname)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        lg.addHandler(ch)
    return lg


class Counters:
    """Named counters for hot loops. Dump with .summary() — logged each tick."""

    def __init__(self):
        self._c = defaultdict(int)
        self._f = defaultdict(float)

    def inc(self, k, n=1):
        self._c[k] += n

    def add(self, k, v):
        self._f[k] += v

    def get(self, k):
        return self._c.get(k, 0) or self._f.get(k, 0)

    def summary(self):
        d = {**{k: v for k, v in self._c.items()}, **{k: round(v, 4) for k, v in self._f.items()}}
        return json.dumps(d, sort_keys=True)


@contextmanager
def timed(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.debug(f"[TIMED] >>> {label}")
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"[TIMED] <<< {label}  ({dt:.3f}s)")


def banner(logger: logging.Logger, text: str):
    bar = "=" * min(72, max(24, len(text) + 8))
    logger.info(bar)
    logger.info(f"=== {text}")
    logger.info(bar)
