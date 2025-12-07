import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
