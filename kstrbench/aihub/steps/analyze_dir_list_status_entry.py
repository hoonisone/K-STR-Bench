from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AnalyzeDirListStatusStep:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        list_file: Any | None = None,
        out_file: Any | None = None,
        show_progress: Any = True,
        sample_limit: Any = 20,
        log_level: Any | None = None,
    ) -> None:
        self.dataset_dir_raw = dataset_dir
        self.list_file_raw = list_file
        self.out_file_raw = out_file
        self.show_progress_raw = show_progress
        self.sample_limit_raw = sample_limit
        self.log_level_raw = log_level

    def run(self) -> dict[str, Any]:
        if not self.dataset_dir_raw:
            raise ValueError("dataset_dir is required")
        if not self.list_file_raw:
            raise ValueError("list_file is required")
        if not self.out_file_raw:
            raise ValueError("out_file is required")

        dataset_dir = Path(str(self.dataset_dir_raw)).expanduser().resolve()
        list_file = Path(str(self.list_file_raw)).expanduser().resolve()
        out_file = Path(str(self.out_file_raw)).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
        if not list_file.is_file():
            raise FileNotFoundError(f"list_file not found: {list_file}")

        show_progress = coerce_bool(self.show_progress_raw, default=True)
        sample_limit = int(self.sample_limit_raw)
        if sample_limit < 0:
            raise ValueError("sample_limit must be >= 0")
        log_level = normalize_log_level(self.log_level_raw, default="INFO")

        with list_file.open("r", encoding="utf-8") as f:
            rows = [line.rstrip("\n") for line in f]

        iter_rows: Any = (
            tqdm(rows, desc="analyze dir list status", unit="line", dynamic_ncols=True)
            if show_progress
            else rows
        )

        total_line_count = 0
        non_empty_line_count = 0
        exists_dir_count = 0
        empty_dir_count = 0
        non_empty_dir_count = 0
        missing_count = 0
        non_dir_count = 0
        empty_samples: list[str] = []
        missing_samples: list[str] = []
        non_dir_samples: list[str] = []

        for raw in iter_rows:
            total_line_count += 1
            line = raw.strip()
            if not line:
                continue
            non_empty_line_count += 1

            rel_or_abs = Path(line.replace("\\", "/"))
            target = (
                (dataset_dir / rel_or_abs).resolve()
                if not rel_or_abs.is_absolute()
                else rel_or_abs.resolve()
            )
            if not target.exists():
                missing_count += 1
                if len(missing_samples) < sample_limit:
                    missing_samples.append(line)
                continue
            if not target.is_dir():
                non_dir_count += 1
                if len(non_dir_samples) < sample_limit:
                    non_dir_samples.append(line)
                continue

            exists_dir_count += 1
            try:
                next(target.iterdir())
                non_empty_dir_count += 1
            except StopIteration:
                empty_dir_count += 1
                if len(empty_samples) < sample_limit:
                    empty_samples.append(line)

        out_lines: list[str] = []
        out_lines.append(f"dataset_dir: {dataset_dir}")
        out_lines.append(f"list_file: {list_file}")
        out_lines.append(f"total_line_count: {total_line_count}")
        out_lines.append(f"non_empty_line_count: {non_empty_line_count}")
        out_lines.append(f"exists_dir_count: {exists_dir_count}")
        out_lines.append(f"empty_dir_count: {empty_dir_count}")
        out_lines.append(f"non_empty_dir_count: {non_empty_dir_count}")
        out_lines.append(f"missing_count: {missing_count}")
        out_lines.append(f"non_dir_count: {non_dir_count}")
        out_lines.append("")
        out_lines.append("[empty_dir_samples]")
        out_lines.extend(empty_samples)
        out_lines.append("")
        out_lines.append("[missing_samples]")
        out_lines.extend(missing_samples)
        out_lines.append("")
        out_lines.append("[non_dir_samples]")
        out_lines.extend(non_dir_samples)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out_lines) + "\n")

        result = {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "list_file": str(list_file),
            "out_file": str(out_file),
            "total_line_count": total_line_count,
            "non_empty_line_count": non_empty_line_count,
            "exists_dir_count": exists_dir_count,
            "empty_dir_count": empty_dir_count,
            "non_empty_dir_count": non_empty_dir_count,
            "missing_count": missing_count,
            "non_dir_count": non_dir_count,
            "sample_limit": sample_limit,
            "show_progress": show_progress,
            "log_level": log_level,
        }

        logger = StepLogger("analyze_dir_list_status_entry")
        logger.log(
            {
                **result,
                "empty_dir_samples": empty_samples,
                "missing_samples": missing_samples,
                "non_dir_samples": non_dir_samples,
            },
            title="analyze dir list status result",
        )
        emit(log_level, "INFO", f"[analyze_dir_status] list_file={list_file}")
        emit(
            log_level,
            "INFO",
            (
                f"[analyze_dir_status] exists={exists_dir_count}, empty={empty_dir_count}, "
                f"missing={missing_count}, non_dir={non_dir_count}"
            ),
        )
        emit(log_level, "INFO", f"[analyze_dir_status] out_file={out_file}")
        return result

