from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class SubtractPathListStep:
    def __init__(
        self,
        source_list_file: Any | None = None,
        remove_list_file: Any | None = None,
        out_list_file: Any | None = None,
        show_progress: Any = True,
        normalize_slashes: Any = True,
        log_level: Any | None = None,
    ) -> None:
        self.source_list_file_raw = source_list_file
        self.remove_list_file_raw = remove_list_file
        self.out_list_file_raw = out_list_file
        self.show_progress_raw = show_progress
        self.normalize_slashes_raw = normalize_slashes
        self.log_level_raw = log_level

    def run(self) -> dict[str, Any]:
        if not self.source_list_file_raw:
            raise ValueError("source_list_file is required")
        if not self.remove_list_file_raw:
            raise ValueError("remove_list_file is required")
        if not self.out_list_file_raw:
            raise ValueError("out_list_file is required")

        source_list_file = Path(str(self.source_list_file_raw)).expanduser().resolve()
        remove_list_file = Path(str(self.remove_list_file_raw)).expanduser().resolve()
        out_list_file = Path(str(self.out_list_file_raw)).expanduser().resolve()
        if not source_list_file.is_file():
            raise FileNotFoundError(f"source_list_file not found: {source_list_file}")
        if not remove_list_file.is_file():
            raise FileNotFoundError(f"remove_list_file not found: {remove_list_file}")

        show_progress = coerce_bool(self.show_progress_raw, default=True)
        normalize_slashes = coerce_bool(self.normalize_slashes_raw, default=True)
        log_level = normalize_log_level(self.log_level_raw, default="INFO")

        def normalize_line(line: str) -> str:
            s = line.strip()
            if normalize_slashes:
                s = s.replace("\\", "/")
            return s

        with source_list_file.open("r", encoding="utf-8") as f:
            source_rows_raw = [line.rstrip("\n") for line in f]
        with remove_list_file.open("r", encoding="utf-8") as f:
            remove_rows_raw = [line.rstrip("\n") for line in f]

        source_rows = [normalize_line(x) for x in source_rows_raw if normalize_line(x)]
        remove_set = {normalize_line(x) for x in remove_rows_raw if normalize_line(x)}

        iter_rows: Any = (
            tqdm(source_rows, desc="subtract path list", unit="line", dynamic_ncols=True)
            if show_progress
            else source_rows
        )
        kept_rows: list[str] = []
        removed_rows: list[str] = []
        for row in iter_rows:
            if row in remove_set:
                removed_rows.append(row)
            else:
                kept_rows.append(row)

        out_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for row in kept_rows:
                f.write(row + "\n")

        removed_samples = removed_rows[:20]
        result = {
            "ok": True,
            "source_list_file": str(source_list_file),
            "remove_list_file": str(remove_list_file),
            "out_list_file": str(out_list_file),
            "normalize_slashes": normalize_slashes,
            "show_progress": show_progress,
            "source_count": len(source_rows),
            "remove_count": len(remove_set),
            "removed_count": len(removed_rows),
            "kept_count": len(kept_rows),
            "log_level": log_level,
        }

        logger = StepLogger("subtract_path_list_entry")
        logger.log(
            {
                **result,
                "removed_samples": removed_samples,
            },
            title="subtract path list result",
        )
        emit(log_level, "INFO", f"[subtract_list] source={source_list_file}")
        emit(log_level, "INFO", f"[subtract_list] remove={remove_list_file}")
        emit(log_level, "INFO", f"[subtract_list] kept={len(kept_rows)}, removed={len(removed_rows)}")
        emit(log_level, "INFO", f"[subtract_list] out={out_list_file}")
        return result

