import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

class JsonExtraFormatter(logging.Formatter):
    """Appends any custom `extra` parameters to the end of the log message."""
    
    # Standard attributes present on built-in LogRecord objects
    RESERVED_ATTRS = {
        'args', 'asctime', 'created', 'filename', 'funcName', 'levelname',
        'levelno', 'lineno', 'module', 'msecs', 'message', 'msg', 'name',
        'pathname', 'process', 'processName', 'relativeCreated', 'stack_info',
        'thread', 'threadName', 'exc_info', 'exc_text', 'taskName'
    }

    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)
        
        # Extract custom extra attributes
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED_ATTRS
        }
        
        if extras:
            extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base_message} | {extra_str}"
            
        return base_message

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logs for console output with safe structured extras."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonExtraFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    logging.getLogger("app").setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("apscheduler").setLevel(level)
