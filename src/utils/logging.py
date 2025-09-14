import logging
import os


def setup_logger(name: str = "qwen_robot", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

