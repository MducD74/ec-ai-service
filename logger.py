import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{extra[label]}] [{level}]: {message}"

# Labels dùng cho http sink - tách riêng khỏi app log
_HTTP_LABELS = {"HTTP"}


def _inject_label(record):
    """Đảm bảo mọi record đều có key 'label' để format string không bị KeyError."""
    record["extra"].setdefault("label", "-")
    return True


def _app_filter(record):
    """App log: nhận tất cả trừ HTTP request log."""
    record["extra"].setdefault("label", "-")
    return record["extra"]["label"] not in _HTTP_LABELS


def _http_filter(record):
    """HTTP log: chỉ nhận HTTP request log."""
    record["extra"].setdefault("label", "-")
    return record["extra"]["label"] in _HTTP_LABELS


logger.remove()

# Console: nhận tất cả (tiện debug)
logger.add(
    sys.stdout,
    level="DEBUG",
    format=LOG_FORMAT,
    filter=_inject_label,
    backtrace=True,
    diagnose=True,
)

# App log: AI-SERVICE, APP, ... — không có HTTP
logger.add(
    LOG_DIR / "app-{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format=LOG_FORMAT,
    filter=_app_filter,
    backtrace=True,
    diagnose=True,
)

# Error log: nhận tất cả label ở mức ERROR trở lên
logger.add(
    LOG_DIR / "app-error-{time:YYYY-MM-DD}.log",
    level="ERROR",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format=LOG_FORMAT,
    filter=_inject_label,
    backtrace=True,
    diagnose=True,
)

# HTTP request log: chỉ nhận HTTP label
logger.add(
    LOG_DIR / "http-{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format=LOG_FORMAT,
    filter=_http_filter,
)

app_log = logger.bind(label="APP")
http_log = logger.bind(label="HTTP")
ai_service_log = logger.bind(label="AI-SERVICE")