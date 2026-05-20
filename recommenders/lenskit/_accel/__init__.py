"""Fallback stubs for LensKit's optional compiled acceleration module."""


class _MissingAccel:
    def __init__(self, *args, **kwargs):
        raise ImportError("LensKit compiled acceleration module is not available.")


def _missing(*args, **kwargs):
    raise ImportError("LensKit compiled acceleration module is not available.")


class _MissingNamespace:
    def __getattr__(self, name):
        return _MissingAccel


class AtomicInt:
    def __init__(self, value=0):
        self.value = value

    def get(self):
        return self.value

    def increment(self, n=1):
        self.value += n
        return self.value


def init_accel_pool(*args, **kwargs):
    return None


FunkSVDTrainer = _MissingAccel
als = _MissingNamespace()
knn = _MissingNamespace()
slim = _MissingNamespace()
data = _MissingNamespace()
