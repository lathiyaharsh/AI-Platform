from loguru import logger
import sys

from app.config.settings import settings


logger.remove()

logger.add(
    sys.stdout,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{message}"
    ),
)

app_logger = logger