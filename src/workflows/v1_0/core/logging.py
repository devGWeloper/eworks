"""로깅 포맷터 + 초기화 설정"""

import json
import logging
import sys
from datetime import datetime


class ConsoleFormatter(logging.Formatter):
    """개발 환경용 콘솔 포맷터"""

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(7)
        name = record.name.rsplit(".", 1)[-1] if "." in record.name else record.name
        message = record.getMessage()

        base = f"{timestamp} | {level} | {name} | {message}"

        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)

        return base


class JSONFormatter(logging.Formatter):
    """운영 환경용 JSON 포맷터"""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        for key in record.__dict__:
            if key not in self._RESERVED and key not in log_data:
                log_data[key] = record.__dict__[key]

        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """앱 시작 시 1회 호출 — 로깅 포맷 설정"""
    formatter = JSONFormatter() if json_output else ConsoleFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
