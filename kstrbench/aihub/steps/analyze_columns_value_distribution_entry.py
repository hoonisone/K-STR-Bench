from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AnalyzeColumnsValueDistributionStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _safe_filename(text: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in text.strip())
    return out or "column"


def _parse_filters(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("filters must be a list")
    parsed: list[dict[str, Any]] = []
    allowed_ops = {"eq", "ne", "in", "not_in"}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"filters[{i}] must be an object")
        column = str(item.get("column", "")).strip()
        op = str(item.get("op", "in")).strip().lower()
        value = item.get("value")
        values = _as_list(item.get("values"))
        if not column:
            raise ValueError(f"filters[{i}].column is required")
        if op not in allowed_ops:
            raise ValueError(f"filters[{i}].op is invalid: {op}")
        if op in ("in", "not_in") and not values:
            raise ValueError(f"filters[{i}].values is required for op={op}")
        if op in ("eq", "ne") and value is None:
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
    return False


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    in_csv_file_raw = params.get("in_csv_file")
    columns = _as_list(params.get("columns"))
    out_dir_raw = params.get("out_dir")
    filters = _parse_filters(params.get("filters"))
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    sort_by_count = coerce_bool(params.get("sort_by_count", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not columns:
        raise ValueError("columns is required")
    if not out_dir_raw:
        raise ValueError("out_dir is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_dir = Path(str(out_dir_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        missing_columns = [col for col in columns if col not in fields]
        if missing_columns:
            raise ValueError(f"CSV missing columns: {missing_columns}")
        for flt in filters:
            if str(flt["column"]) not in fields:
                raise ValueError(f"CSV missing filters.column '{flt['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    filtered_rows = rows
    for flt in filters:
        col = str(flt["column"])
        filtered_rows = [
            r
            for r in filtered_rows
            if (
                (not bool(flt.get("exclude", False)) and _evaluate_filter(r.get(col, ""), flt))
                or (bool(flt.get("exclude", False)) and (not _evaluate_filter(r.get(col, ""), flt)))
            )
        ]

    counters: dict[str, Counter[str]] = {col: Counter() for col in columns}
    row_iter: Any = (
        tqdm(filtered_rows, desc="analyze columns value distribution", unit="row", dynamic_ncols=True)
        if show_progress
        else filtered_rows
    )
    for row in row_iter:
        for col in columns:
            raw = row.get(col, "")
            val = str(raw) if raw is not None else ""
            if val == "":
                val = "<EMPTY>"
            counters[col][val] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files: dict[str, str] = {}
    for col in columns:
        out_file = (out_dir / f"{_safe_filename(col)}.csv").resolve()
        items = list(counters[col].items())
        if sort_by_count:
            items.sort(key=lambda x: (-x[1], x[0]))
        else:
            items.sort(key=lambda x: x[0])
        with out_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value", "count"])
            for value, cnt in items:
                writer.writerow([value, int(cnt)])
        out_files[col] = str(out_file)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_dir": str(out_dir),
        "columns": columns,
        "filters": filters,
        "input_row_count": len(rows),
        "row_count": len(filtered_rows),
        "out_files": out_files,
        "show_progress": show_progress,
        "sort_by_count": sort_by_count,
        "log_level": log_level,
    }
    logger = StepLogger("analyze_columns_value_distribution_entry")
    logger.log(summary, title="analyze columns value distribution summary")
    emit(log_level, "INFO", f"[col-value-dist] rows={len(filtered_rows)}, columns={len(columns)}")
    emit(log_level, "INFO", f"[col-value-dist] out_dir={out_dir}")
    return summary

