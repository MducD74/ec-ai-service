# logger.py
import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()

# Console log khi dev
logger.add(
    sys.stdout,
    level="DEBUG",
    format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{extra[label]}] [{level}]: {message}",
    backtrace=True,
    diagnose=True,
)

# App log
logger.add(
    LOG_DIR / "app-{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{extra[label]}] [{level}]: {message}",
    backtrace=True,
    diagnose=True,
)

# Error log riêng
logger.add(
    LOG_DIR / "app-error-{time:YYYY-MM-DD}.log",
    level="ERROR",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{extra[label]}] [{level}]: {message}",
    backtrace=True,
    diagnose=True,
)

# HTTP request log
logger.add(
    LOG_DIR / "http-{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{extra[label]}] [{level}]: {message}",
)

app_log = logger.bind(label="APP")
http_log = logger.bind(label="HTTP")
ai_service_log = logger.bind(label="AI-SERVICE")