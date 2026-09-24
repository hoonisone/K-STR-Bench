from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class DropDuplicateTextIdsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    text_id_col = str(params.get("text_id_col", "text_id")).strip()
    keep = str(params.get("keep", "first")).strip().lower()
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not text_id_col:
        raise ValueError("text_id_col is required")
    if keep != "first":
        raise ValueError("keep must be 'first'")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("drop_duplicate_text_ids_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "text_id_col": text_id_col,
            "keep": keep,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="drop duplicate text ids discovery",
    )

    tmp_csv = out_csv_file.with_name(out_csv_file.name + ".tmp")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    input_row_count = 0
    kept_row_count = 0
    removed_row_count = 0

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        if text_id_col not in fieldnames:
            raise ValueError(f"CSV missing column '{text_id_col}': {in_csv_file}")

        row_iter: Any = (
            tqdm(reader, desc="drop duplicate text ids", unit="row", dynamic_ncols=True)
            if show_progress
            else reader
        )
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in row_iter:
                input_row_count += 1
                text_id = str(row.get(text_id_col, "") or "").strip()
                if text_id:
                    if text_id in seen:
                        removed_row_count += 1
                        continue
                    seen.add(text_id)
                writer.writerow(row)
                kept_row_count += 1

    tmp_csv.replace(out_csv_file)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "text_id_col": text_id_col,
        "keep": keep,
        "input_row_count": input_row_count,
        "kept_row_count": kept_row_count,
        "removed_row_count": removed_row_count,
        "unique_text_id_count": len(seen),
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(summary, title="drop duplicate text ids summary")
    emit(
        log_level,
        "INFO",
        (
            f"[drop-text-id] rows={input_row_count}, kept={kept_row_count}, "
            f"removed={removed_row_count}, unique={len(seen)}"
        ),
    )
    emit(log_level, "INFO", f"[drop-text-id] out_csv={out_csv_file}")
    return summary
