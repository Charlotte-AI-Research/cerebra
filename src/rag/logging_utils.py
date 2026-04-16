from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_CONFIGURED = False


@dataclass(frozen=True)
class LogConfig:
    level: str
    log_file: Optional[str]
    json: bool


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_log_config() -> LogConfig:
    return LogConfig(
        level=os.getenv("CEREBRA_LOG_LEVEL", "INFO").upper(),
        log_file=os.getenv("CEREBRA_LOG_FILE"),
        json=_env_bool("CEREBRA_LOG_JSON", default=False),
    )


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(*, name: str = "cerebra") -> logging.Logger:
    """
    Configure logging once for the process.

    Env:
      - CEREBRA_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
      - CEREBRA_LOG_FILE: path to file (optional)
      - CEREBRA_LOG_JSON: true/false (default: false)
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    cfg = get_log_config()
    level = getattr(logging, cfg.level, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    formatter: logging.Formatter
    formatter = _JsonFormatter() if cfg.json else logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if cfg.log_file:
        log_path = Path(cfg.log_file).expanduser()
    else:
        # Default to repo-local log file so it works in containers/servers.
        log_path = Path.cwd() / "cerebra.log"

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.debug("Logging configured", extra={"extra": {"log_file": str(log_path), "level": cfg.level}})
    except Exception:
        # If file logging fails, keep console logging alive.
        logger.warning("Failed to enable file logging", exc_info=True)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    base = setup_logging()
    return base.getChild(name)

