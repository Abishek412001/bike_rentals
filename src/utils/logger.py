import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_file: str = "app.log"):
    """
    Configures Loguru structured logging for standard output and file logging.
    """
    logger.remove()  # Remove default handler

    # Console Handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File Handler
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.add(
        logs_dir / log_file,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    return logger


app_logger = setup_logger()
