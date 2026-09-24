from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AnalyzeColumnDuplicatesStep:
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


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    target_columns = _as_str_list(params.get("target_columns", params.get("columns")))
    out_report_file_raw = params.get("out_report_file")
    out_csv_file_raw = params.get("out_csv_file")
    drop_duplicate = coerce_bool(
        params.get("drop_duplicate", params.get("drop_dupplicate", False)),
        default=False,
    )
    check_duplicate = coerce_bool(params.get("check_duplicate", False), default=False)
    is_valid_col_name = str(params.get("is_valid_col_name", "is_valid")).strip()
    skip_empty = coerce_bool(params.get("skip_empty", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not target_columns:
        raise ValueError("target_columns (or columns) is required")
    if not out_report_file_raw:
        raise ValueError("out_report_file is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file is required")
    if check_duplicate and not is_valid_col_name:
        raise ValueError("is_valid_col_name is required when check_duplicate is True")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_report_file = Path(str(out_report_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing_columns = [c for c in target_columns if c not in fieldnames]
        if missing_columns:
            raise ValueError(f"CSV missing target columns: {missing_columns}")
        rows = [dict(r) for r in reader]

    summary_rows: list[dict[str, Any]] = []
    col_iter: Any = (
        tqdm(target_columns, desc="analyze column duplicates", unit="column", dynamic_ncols=True)
        if show_progress
        else target_columns
    )
    for col in col_iter:
        counts: dict[str, int] = {}
        non_empty_count = 0
        for row in rows:
            value = str(row.get(col, "") or "")
            if not value.strip():
                if skip_empty:
                    continue
            else:
                non_empty_count += 1
            counts[value] = counts.get(value, 0) + 1

        duplicate_value_count = sum(1 for c in counts.values() if c > 1)
        duplicate_row_count = sum(c - 1 for c in counts.values() if c > 1)
        unique_value_count = len(counts)
        summary_rows.append(
            {
                "column": col,
                "row_count": len(rows),
                "non_empty_count": non_empty_count,
                "unique_value_count": unique_value_count,
                "duplicate_value_count": duplicate_value_count,
                "duplicate_row_count": duplicate_row_count,
                "is_unique": duplicate_row_count == 0,
            }
        )

    out_report_file.parent.mkdir(parents=True, exist_ok=True)
    with out_report_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "column",
                "row_count",
                "non_empty_count",
                "unique_value_count",
                "duplicate_value_count",
                "duplicate_row_count",
                "is_unique",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    processed_rows: list[dict[str, Any]] = [dict(r) for r in rows]
    processed_fields = list(fieldnames)

    dedup_removed_count = 0
    if drop_duplicate:
        seen_keys: set[tuple[str, ...]] = set()
        dedup_rows: list[dict[str, Any]] = []
        for row in processed_rows:
            key = tuple(str(row.get(col, "") or "") for col in target_columns)
            if key in seen_keys:
                dedup_removed_count += 1
                continue
            seen_keys.add(key)
            dedup_rows.append(row)
        processed_rows = dedup_rows

    check_true_count = 0
    check_false_count = 0
    if check_duplicate:
        key_counts: dict[tuple[str, ...], int] = {}
        for row in processed_rows:
            key = tuple(str(row.get(col, "") or "") for col in target_columns)
            if skip_empty and (not any(v.strip() for v in key)):
                continue
            key_counts[key] = key_counts.get(key, 0) + 1

        check_rows: list[dict[str, Any]] = []
        for row in processed_rows:
            key = tuple(str(row.get(col, "") or "") for col in target_columns)
            is_duplicate = key_counts.get(key, 0) > 1
            is_valid = not is_duplicate
            if is_valid:
                check_true_count += 1
            else:
                check_false_count += 1
            out_row = dict(row)
            out_row[is_valid_col_name] = is_valid
            check_rows.append(out_row)
        processed_rows = check_rows
        if is_valid_col_name not in processed_fields:
            processed_fields.append(is_valid_col_name)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=processed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(processed_rows)

    total_duplicate_row_count = sum(int(r["duplicate_row_count"]) for r in summary_rows)
    total_duplicate_value_count = sum(int(r["duplicate_value_count"]) for r in summary_rows)
    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "target_columns": target_columns,
        "out_report_file": str(out_report_file),
        "out_csv_file": str(out_csv_file),
        "drop_duplicate": drop_duplicate,
        "check_duplicate": check_duplicate,
        "is_valid_col_name": is_valid_col_name if check_duplicate else None,
        "processed_row_count": len(processed_rows),
        "check_true_count": check_true_count,
        "check_false_count": check_false_count,
        "dedup_removed_count": dedup_removed_count,
        "row_count": len(rows),
        "column_count": len(target_columns),
        "total_duplicate_value_count": total_duplicate_value_count,
        "total_duplicate_row_count": total_duplicate_row_count,
        "skip_empty": skip_empty,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger = StepLogger("analyze_column_duplicates_entry")
    logger.log(summary, title="analyze column duplicates summary")
    emit(
        log_level,
        "INFO",
        (
            f"[col-dup] columns={len(target_columns)}, rows={len(rows)}, "
            f"dup_values={total_duplicate_value_count}, dup_rows={total_duplicate_row_count}, "
            f"dedup_removed={dedup_removed_count}, check_true={check_true_count}"
        ),
    )
    emit(log_level, "INFO", f"[col-dup] out_report={out_report_file}")
    emit(log_level, "INFO", f"[col-dup] out_csv={out_csv_file}")
    return summary
