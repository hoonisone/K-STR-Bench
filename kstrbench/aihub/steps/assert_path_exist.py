from __future__ import annotations

from pathlib import Path
from typing import Any

from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AssertPathExist:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        list_file: Any | None = None,
        error_message: Any | None = None,
        max_print_missing: Any = 10,
        log_level: Any | None = None,
    ) -> None:
        self.dataset_dir_raw = dataset_dir
        self.list_file_raw = list_file
        self.error_message_raw = error_message
        self.max_print_missing_raw = max_print_missing
        self.log_level_raw = log_level

    def run(self) -> dict[str, Any]:
        if not self.dataset_dir_raw:
            raise ValueError("dataset_dir is required")
        if not self.list_file_raw:
            raise ValueError("list_file is required")

        dataset_dir = Path(str(self.dataset_dir_raw)).expanduser().resolve()
        list_file = Path(str(self.list_file_raw)).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
        if not list_file.is_file():
            raise FileNotFoundError(f"list_file not found: {list_file}")

        log_level = normalize_log_level(self.log_level_raw, default="INFO")
        max_print_missing = int(self.max_print_missing_raw)
        if max_print_missing < 0:
            raise ValueError("max_print_missing must be >= 0")
        error_message = (
            str(self.error_message_raw)
            if self.error_message_raw is not None
            else "One or more paths do not exist."
        )

        total_count = 0
        existing_count = 0
        missing_paths: list[str] = []
        with list_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                total_count += 1
                rel_or_abs = Path(line.replace("\\", "/"))
                target = (dataset_dir / rel_or_abs).resolve() if not rel_or_abs.is_absolute() else rel_or_abs.resolve()
                if target.exists():
                    existing_count += 1
                else:
                    missing_paths.append(str(target))

        missing_count = len(missing_paths)
        is_ok = missing_count == 0

        logger = StepLogger("assert_path_exist")
        logger.log(
            {
                "dataset_dir": str(dataset_dir),
                "list_file": str(list_file),
                "total_count": total_count,
                "existing_count": existing_count,
                "missing_count": missing_count,
                "missing_sample": missing_paths[:max_print_missing],
                "max_print_missing": max_print_missing,
                "is_ok": is_ok,
                "log_level": log_level,
            },
            title="assert path exist result",
        )

        if not is_ok:
            emit(log_level, "ERROR", f"[assert_path_exist] missing={missing_count}/{total_count}")
            raise RuntimeError(error_message)

        emit(log_level, "INFO", f"[assert_path_exist] all paths exist: {existing_count}/{total_count}")
        return {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "list_file": str(list_file),
            "total_count": total_count,
            "existing_count": existing_count,
            "missing_count": 0,
            "log_level": log_level,
        }
