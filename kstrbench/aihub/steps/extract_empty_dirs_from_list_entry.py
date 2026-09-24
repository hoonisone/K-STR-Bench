from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class ExtractEmptyDirsFromListStep:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        source_list_file: Any | None = None,
        out_empty_list_file: Any | None = None,
        drop_filtered_file_from_source_list: Any = False,
        updated_source_file: Any | None = None,
        include_missing_paths: Any = False,
        show_progress: Any = True,
        log_level: Any | None = None,
    ) -> None:
        self.dataset_dir_raw = dataset_dir
        self.source_list_file_raw = source_list_file
        self.out_empty_list_file_raw = out_empty_list_file
        self.drop_filtered_file_from_source_list_raw = drop_filtered_file_from_source_list
        self.updated_source_file_raw = updated_source_file
        self.include_missing_paths_raw = include_missing_paths
        self.show_progress_raw = show_progress
        self.log_level_raw = log_level

    def run(self) -> dict[str, Any]:
        if not self.dataset_dir_raw:
            raise ValueError("dataset_dir is required")
        if not self.source_list_file_raw:
            raise ValueError("source_list_file is required")
        if not self.out_empty_list_file_raw:
            raise ValueError("out_empty_list_file is required")

        dataset_dir = Path(str(self.dataset_dir_raw)).expanduser().resolve()
        source_list_file = Path(str(self.source_list_file_raw)).expanduser().resolve()
        out_empty_list_file = Path(str(self.out_empty_list_file_raw)).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
        if not source_list_file.is_file():
            raise FileNotFoundError(f"source_list_file not found: {source_list_file}")

        drop_filtered = coerce_bool(
            self.drop_filtered_file_from_source_list_raw,
            default=False,
        )
        updated_source_file = (
            Path(str(self.updated_source_file_raw)).expanduser().resolve()
            if self.updated_source_file_raw is not None
            else None
        )
        if drop_filtered and updated_source_file is None:
            raise ValueError(
                "updated_source_file is required when drop_filtered_file_from_source_list=true"
            )

        show_progress = coerce_bool(self.show_progress_raw, default=True)
        include_missing_paths = coerce_bool(self.include_missing_paths_raw, default=False)
        log_level = normalize_log_level(self.log_level_raw, default="INFO")

        with source_list_file.open("r", encoding="utf-8") as f:
            source_rows = [line.rstrip("\n") for line in f]

        iter_rows: Any = (
            tqdm(source_rows, desc="extract empty dirs", unit="line", dynamic_ncols=True)
            if show_progress
            else source_rows
        )

        total_line_count = 0
        non_empty_line_count = 0
        empty_dir_lines: list[str] = []
        kept_lines: list[str] = []
        missing_path_count = 0
        non_dir_path_count = 0
        non_empty_dir_count = 0

        for raw_line in iter_rows:
            total_line_count += 1
            line = raw_line.strip()
            if not line:
                kept_lines.append("")
                continue
            non_empty_line_count += 1
            rel_or_abs = Path(line.replace("\\", "/"))
            target = (
                (dataset_dir / rel_or_abs).resolve()
                if not rel_or_abs.is_absolute()
                else rel_or_abs.resolve()
            )

            if not target.exists():
                missing_path_count += 1
                if include_missing_paths:
                    empty_dir_lines.append(line)
                else:
                    kept_lines.append(line)
                continue
            if not target.is_dir():
                non_dir_path_count += 1
                kept_lines.append(line)
                continue

            is_empty = False
            try:
                next(target.iterdir())
                is_empty = False
            except StopIteration:
                is_empty = True

            if is_empty:
                empty_dir_lines.append(line)
            else:
                non_empty_dir_count += 1
                kept_lines.append(line)

        out_empty_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_empty_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for line in empty_dir_lines:
                f.write(line + "\n")

        updated_source_count: int | None = None
        if drop_filtered and updated_source_file is not None:
            updated_source_file.parent.mkdir(parents=True, exist_ok=True)
            with updated_source_file.open("w", encoding="utf-8", newline="\n") as f:
                for line in kept_lines:
                    f.write(line + "\n")
            updated_source_count = len(kept_lines)

        result = {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "source_list_file": str(source_list_file),
            "out_empty_list_file": str(out_empty_list_file),
            "drop_filtered_file_from_source_list": drop_filtered,
            "updated_source_file": str(updated_source_file) if updated_source_file else None,
            "updated_source_count": updated_source_count,
            "include_missing_paths": include_missing_paths,
            "total_line_count": total_line_count,
            "non_empty_line_count": non_empty_line_count,
            "empty_dir_count": len(empty_dir_lines),
            "non_empty_dir_count": non_empty_dir_count,
            "missing_path_count": missing_path_count,
            "non_dir_path_count": non_dir_path_count,
            "show_progress": show_progress,
            "log_level": log_level,
        }

        logger = StepLogger("extract_empty_dirs_from_list_entry")
        logger.log(
            {
                **result,
                "empty_dir_samples": empty_dir_lines[:20],
            },
            title="extract empty dirs result",
        )

        emit(log_level, "INFO", f"[empty_dirs] source={source_list_file}")
        emit(log_level, "INFO", f"[empty_dirs] out_empty_list_file={out_empty_list_file}")
        emit(
            log_level,
            "INFO",
            (
                f"[empty_dirs] empty={len(empty_dir_lines)}, non_empty={non_empty_dir_count}, "
                f"missing={missing_path_count}, non_dir={non_dir_path_count}"
            ),
        )
        if drop_filtered and updated_source_file is not None:
            emit(log_level, "INFO", f"[empty_dirs] updated_source_file={updated_source_file}")

        return result

