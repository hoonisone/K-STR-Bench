from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _sanitize_step_name(step_name: str) -> str:
    cleaned = re.sub(r"[^\w\-\.]+", "_", step_name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed_step"


def _default_logs_dir() -> Path:
    from kstrbench.dataset_dir import aihub_standardized, dataset_root

    return aihub_standardized(dataset_root()) / "logs"


def _to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return str(payload)


def normalize_log_level(value: Any, default: str = "INFO") -> str:
    if value is None:
        return default
    level = str(value).strip().upper()
    if level == "WARN":
        level = "WARNING"
    if level == "FATAL":
        level = "CRITICAL"
    if level not in LOG_LEVELS:
        allowed = ", ".join(LOG_LEVELS.keys())
        raise ValueError(f"invalid log_level: {value} (allowed: {allowed})")
    return level


def should_emit(config_level: str, message_level: str) -> bool:
    config_norm = normalize_log_level(config_level)
    message_norm = normalize_log_level(message_level)
    return LOG_LEVELS[message_norm] >= LOG_LEVELS[config_norm]


def emit(config_level: str, message_level: str, message: str) -> None:
    if should_emit(config_level, message_level):
        print(message)


class StepLogger:
    """
    생성 시 파일명을 확정하고, 이후 모든 로그를 같은 파일에 append한다.

    파일명 형식:
    - {YYYYMMDD_HHMMSS}_{step_name}.txt
    - 동일 파일이 있으면 _2, _3 ... suffix를 붙여 저장
    """

    def __init__(
        self,
        step_name: str,
        logs_dir: Path | str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.step_name = step_name
        self.step_safe = _sanitize_step_name(step_name)
        self.created_at = timestamp or datetime.now()
        self.base_dir = (
            Path(logs_dir).expanduser().resolve() if logs_dir else _default_logs_dir()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        ts = self.created_at.strftime("%Y%m%d_%H%M%S")
        ms = f"{self.created_at.microsecond // 1000:03d}"
        base_name = f"{ts}_{self.step_safe}"
        base_name = f"{ts}_{ms}_{self.step_safe}"
        self.log_path = self.base_dir / f"{base_name}.txt"
        suffix = 2
        while self.log_path.exists():
            self.log_path = self.base_dir / f"{base_name}_{suffix}.txt"
            suffix += 1

        # 파일 헤더를 최초 1회 기록
        header_lines = [
            f"logger_created_at: {self.created_at.isoformat(timespec='seconds')}",
            f"step_name: {self.step_name}",
            "",
        ]
        with self.log_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(header_lines))

    def log(self, payload: Any, title: str | None = None) -> Path:
        now = datetime.now().isoformat(timespec="seconds")
        body = _to_text(payload)
        lines = []
        lines.append(f"[{now}]")
        if title:
            lines.append(f"title: {title}")
        lines.append(body)
        lines.append("")
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        return self.log_path


def log_step_result(
    step_name: str,
    result: Any,
    logs_dir: Path | str | None = None,
) -> Path:
    """
    단발성(one-shot) 기록용 래퍼.
    """
    logger = StepLogger(step_name=step_name, logs_dir=logs_dir)
    return logger.log(result, title="result")

