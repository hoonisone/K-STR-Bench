from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class ExtractBboxSizeColumnsStep:
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


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    value = raw
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            value = json.loads(s)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        if all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in value[:4]):
            xs = [float(v[0]) for v in value[:4]]
            ys = [float(v[1]) for v in value[:4]]
            left = min(xs)
            top = min(ys)
            return left, top, max(xs) - left, max(ys) - top
        x = float(value[0])
        y = float(value[1])
        w = float(value[2])
        h = float(value[3])
        return x, y, w, h
    except (TypeError, ValueError):
        return None


def _format_number(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    bbox_col = str(params.get("bbox_col", "bbox")).strip()
    width_col = str(params.get("width_col", "width")).strip()
    height_col = str(params.get("height_col", "height")).strip()
    filters = _parse_filters(params.get("filters"))
    overwrite_existing = coerce_bool(params.get("overwrite_existing", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not bbox_col:
        raise ValueError("bbox_col is required")
    if not width_col:
        raise ValueError("width_col is required")
    if not height_col:
        raise ValueError("height_col is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("extract_bbox_size_columns_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "bbox_col": bbox_col,
            "width_col": width_col,
            "height_col": height_col,
            "filters": filters,
            "overwrite_existing": overwrite_existing,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="extract bbox size columns discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        if bbox_col not in fieldnames:
            raise ValueError(f"CSV missing bbox_col '{bbox_col}': {in_csv_file}")
        for flt in filters:
            if str(flt["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{flt['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="extract bbox size columns", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    extracted_count = 0
    skipped_by_filter_count = 0
    skipped_existing_count = 0
    for row_index, row in enumerate(row_iter):
        matched = all(
            (not _evaluate_filter(row.get(str(f["column"]), ""), f))
            if bool(f.get("exclude", False))
            else _evaluate_filter(row.get(str(f["column"]), ""), f)
            for f in filters
        )
        if not matched:
            skipped_by_filter_count += 1
            continue

        if not overwrite_existing:
            has_width = str(row.get(width_col, "") or "").strip() != ""
            has_height = str(row.get(height_col, "") or "").strip() != ""
            if has_width or has_height:
                skipped_existing_count += 1
                continue

        bbox = _parse_bbox(row.get(bbox_col))
        if bbox is None:
            raise ValueError(f"invalid bbox at row_index={row_index}: {bbox_col}={row.get(bbox_col)!r}")
        _, _, w, h = bbox
        row[width_col] = _format_number(w)
        row[height_col] = _format_number(h)
        extracted_count += 1

    out_fields = list(fieldnames)
    if width_col not in out_fields:
        out_fields.append(width_col)
    if height_col not in out_fields:
        out_fields.append(height_col)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "bbox_col": bbox_col,
        "width_col": width_col,
        "height_col": height_col,
        "filters": filters,
        "overwrite_existing": overwrite_existing,
        "row_count": len(rows),
        "extracted_count": extracted_count,
        "skipped_by_filter_count": skipped_by_filter_count,
        "skipped_existing_count": skipped_existing_count,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(summary, title="extract bbox size columns summary")
    emit(
        log_level,
        "INFO",
        (
            f"[bbox-size] rows={len(rows)}, extracted={extracted_count}, "
            f"skipped_filter={skipped_by_filter_count}, skipped_existing={skipped_existing_count}"
        ),
    )
    emit(log_level, "INFO", f"[bbox-size] out_csv={out_csv_file}")
    return summary
