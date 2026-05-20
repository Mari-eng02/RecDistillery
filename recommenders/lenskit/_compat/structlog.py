import logging


class BoundLogger:
    def __init__(self, logger=None):
        self._logger = logger or logging.getLogger()

    def bind(self, **kwargs):
        return self

    def unbind(self, *args):
        return self

    def new(self, **kwargs):
        return self

    def debug(self, event=None, **kwargs):
        self._logger.debug(event or "", extra={})

    def info(self, event=None, **kwargs):
        self._logger.info(event or "", extra={})

    def warning(self, event=None, **kwargs):
        self._logger.warning(event or "", extra={})

    warn = warning

    def error(self, event=None, **kwargs):
        self._logger.error(event or "", extra={})

    def exception(self, event=None, **kwargs):
        self._logger.exception(event or "", extra={})


class _Stdlib:
    BoundLogger = BoundLogger

    @staticmethod
    def get_logger(name=None):
        return BoundLogger(logging.getLogger(name))


stdlib = _Stdlib()


class DropEvent(Exception):
    pass


def get_logger(name=None):
    return BoundLogger(logging.getLogger(name))


def make_filtering_bound_logger(level):
    return BoundLogger


def is_configured():
    return False
