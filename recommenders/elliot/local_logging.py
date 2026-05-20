import logging


def _logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return logger


def get_logger(name, level=logging.DEBUG):
    return _logger(name, level)


def get_logger_model(name, level=logging.DEBUG):
    return _logger(name, level)
