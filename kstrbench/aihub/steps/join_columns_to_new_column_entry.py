from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class JoinColumnsToNewColumnStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _as_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _dedupe_values_with_suffix(rows: list[dict[str, str]], column: str) -> int:
    used_values: set[str] = set()
    base_seen_counts: dict[str, int] = {}
    deduped_count = 0
    for row in rows:
        base_value = str(row.get(column, "") or "").strip()
        if not base_value:
            continue
        if base_value not in used_values:
            used_values.add(base_value)
            base_seen_counts[base_value] = 0
            continue
        next_index = int(base_seen_counts.get(base_value, 0)) + 1
        candidate = f"{base_value}_{next_index}"
        while candidate in used_values:
            next_index += 1
            candidate = f"{base_value}_{next_index}"
        row[column] = candidate
        used_values.add(candidate)
        base_seen_counts[base_value] = next_index
        deduped_count += 1
    return deduped_count


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    source_columns = _as_str_list(params.get("source_columns"))
    out_column = str(params.get("out_column", "")).strip()
    join_delimiter = str(params.get("join_delimiter", "_"))
    overwrite_existing = coerce_bool(params.get("overwrite_existing", True), default=True)
    dedupe_annotation_id = coerce_bool(params.get("dedupe_annotation_id", False), default=False)
    strict_columns = coerce_bool(params.get("strict_columns", True), default=True)
    fill_missing_with_empty = coerce_bool(params.get("fill_missing_with_empty", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not source_columns:
        raise ValueError("source_columns is required")
    if not out_column:
        raise ValueError("out_column is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        missing_columns = [c for c in source_columns if c not in fieldnames]
        if strict_columns and missing_columns:
            raise ValueError(f"CSV missing source columns: {missing_columns}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="join columns to new column", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    updated_count = 0
    skipped_count = 0
    missing_cell_count = 0
    for row in row_iter:
        if (not overwrite_existing) and str(row.get(out_column, "") or "").strip():
            skipped_count += 1
            continue

        parts: list[str] = []
        missing_on_row = False
        for col in source_columns:
            if col not in row:
                missing_on_row = True
                parts.append("")
                continue
            value = row.get(col)
            text = "" if value is None else str(value).strip()
            if not text:
                missing_on_row = True
            parts.append(text)

        if missing_on_row:
            missing_cell_count += 1
            if not fill_missing_with_empty:
                skipped_count += 1
                continue

        row[out_column] = join_delimiter.join(parts)
        updated_count += 1

    deduped_count = _dedupe_values_with_suffix(rows, out_column) if dedupe_annotation_id else 0

    out_fields = list(fieldnames)
    if out_column not in out_fields:
        out_fields.append(out_column)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "source_columns": source_columns,
        "out_column": out_column,
        "join_delimiter": join_delimiter,
        "overwrite_existing": overwrite_existing,
        "dedupe_annotation_id": dedupe_annotation_id,
        "strict_columns": strict_columns,
        "fill_missing_with_empty": fill_missing_with_empty,
        "row_count": len(rows),
        "updated_count": updated_count,
        "deduped_count": deduped_count,
        "skipped_count": skipped_count,
        "missing_cell_count": missing_cell_count,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger = StepLogger("join_columns_to_new_column_entry")
    logger.log(summary, title="join columns to new column summary")
    emit(log_level, "INFO", f"[join-cols] rows={len(rows)}, updated={updated_count}, skipped={skipped_count}")
    emit(log_level, "INFO", f"[join-cols] out_csv={out_csv_file}")
    return summary
