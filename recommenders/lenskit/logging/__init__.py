"""Minimal logging helpers for standalone LensKit model definitions."""

from ._stopwatch import Stopwatch


def get_logger(name=None):
    from recommenders.lenskit._compat.structlog import get_logger as _get_logger

    return _get_logger(name)


class Tracer:
    def __init__(self, *args, **kwargs):
        pass

    def span(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def get_tracer(*args, **kwargs):
    return Tracer()


def trace(*args, **kwargs):
    return Tracer()


class Progress:
    def __init__(self, *args, **kwargs):
        self.count = 0

    def update(self, n=1, **kwargs):
        self.count += n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def item_progress(*args, **kwargs):
    return Progress()


def set_progress_impl(*args, **kwargs):
    return None


def friendly_duration(value):
    return str(value)
