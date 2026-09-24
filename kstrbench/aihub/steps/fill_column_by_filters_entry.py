from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class FillColumnByFiltersStep:
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


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_filters(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("filters must be a list")
    parsed: list[dict[str, Any]] = []
    allowed_ops = {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "contains", "not_contains"}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"filters[{i}] must be an object")
        column = str(item.get("column", "")).strip()
        op = str(item.get("op", "in")).strip().lower()
        value = item.get("value")
        values = _as_str_list(item.get("values"))
        if not column:
            raise ValueError(f"filters[{i}].column is required")
        if op not in allowed_ops:
            raise ValueError(f"filters[{i}].op is invalid: {op}")
        if op in ("in", "not_in") and not values:
            raise ValueError(f"filters[{i}].values is required for op={op}")
        if op in ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains") and value is None:
            raise ValueError(f"filters[{i}].value is required for op={op}")
        exclude = coerce_bool(item.get("exclude", False), default=False)
        parsed.append({"column": column, "op": op, "value": value, "values": values, "exclude": exclude})
    return parsed


def _evaluate_filter(raw_value: Any, filt: dict[str, Any]) -> bool:
    op = str(filt.get("op", "in"))
    text = str(raw_value) if raw_value is not None else ""
    if op == "in":
        return text in {str(v) for v in filt.get("values", [])}
    if op == "not_in":
        return text not in {str(v) for v in filt.get("values", [])}
    if op == "eq":
        return text == str(filt.get("value"))
    if op == "ne":
        return text != str(filt.get("value"))
    if op == "contains":
        return str(filt.get("value")) in text
    if op == "not_contains":
        return str(filt.get("value")) not in text
    if op in ("gt", "gte", "lt", "lte"):
        left = _to_float(raw_value)
        right = _to_float(filt.get("value"))
        if left is None or right is None:
            return False
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right
    return False


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    out_column = str(params.get("out_column", "")).strip()
    filters = _parse_filters(params.get("filters"))
    result_value = params.get("result_value")
    default_value = params.get("default_value")
    source_column = str(params.get("source_column", "")).strip()
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not out_column:
        raise ValueError("out_column is required")
    if not filters:
        raise ValueError("filters is required")
    if default_value is None:
        raise ValueError("default_value is required")
    if not source_column and result_value is None:
        raise ValueError("result_value is required when source_column is not provided")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for flt in filters:
            if str(flt["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{flt['column']}': {in_csv_file}")
        if source_column and source_column not in fieldnames:
            raise ValueError(f"CSV missing source_column '{source_column}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="fill column by filters", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    matched_count = 0
    default_count = 0
    result_value_str = "" if result_value is None else str(result_value)
    default_value_str = str(default_value)
    for row in row_iter:
        matched = all(
            (not _evaluate_filter(row.get(str(f["column"]), ""), f))
            if bool(f.get("exclude", False))
            else _evaluate_filter(row.get(str(f["column"]), ""), f)
            for f in filters
        )
        if matched:
            matched_count += 1
            row[out_column] = str(row.get(source_column, "")) if source_column else result_value_str
        else:
            default_count += 1
            row[out_column] = default_value_str

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
        "out_column": out_column,
        "filters": filters,
        "source_column": source_column or None,
        "result_value": None if source_column else result_value_str,
        "default_value": default_value_str,
        "row_count": len(rows),
        "matched_count": matched_count,
        "default_count": default_count,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger = StepLogger("fill_column_by_filters_entry")
    logger.log(summary, title="fill column by filters summary")
    emit(log_level, "INFO", f"[fill-col] rows={len(rows)}, matched={matched_count}, default={default_count}")
    emit(log_level, "INFO", f"[fill-col] out_csv={out_csv_file}")
    return summary

