from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AddTextLengthColumnStep:
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
        values = _as_str_list(item.get("values"))
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

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    text_column = str(params.get("text_column", params.get("label_column", "text"))).strip()
    out_column = str(params.get("out_column", "text_length")).strip()
    filters = _parse_filters(params.get("filters"))
    ignore_space = coerce_bool(params.get("ignore_space", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not text_column:
        raise ValueError("text_column (or label_column) is required")
    if not out_column:
        raise ValueError("out_column is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if text_column not in fieldnames:
            raise ValueError(f"CSV missing text_column '{text_column}': {in_csv_file}")
        for f in filters:
            if str(f["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{f['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    filtered_rows = rows
    for f in filters:
        col = str(f["column"])
        filtered_rows = [
            r
            for r in filtered_rows
            if (
                (not bool(f.get("exclude", False)) and _evaluate_filter(r.get(col, ""), f))
                or (bool(f.get("exclude", False)) and (not _evaluate_filter(r.get(col, ""), f)))
            )
        ]

    row_iter: Any = (
        tqdm(filtered_rows, desc="add text length column", unit="row", dynamic_ncols=True)
        if show_progress
        else filtered_rows
    )

    for row in row_iter:
        text = str(row.get(text_column, "") or "")
        if ignore_space:
            text = "".join(ch for ch in text if not ch.isspace())
        row[out_column] = str(len(text))

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
        "text_column": text_column,
        "out_column": out_column,
        "filters": filters,
        "ignore_space": ignore_space,
        "input_row_count": len(rows),
        "row_count": len(filtered_rows),
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger = StepLogger("add_text_length_column_entry")
    logger.log(summary, title="add text length column summary")
    emit(log_level, "INFO", f"[text-len-col] rows={len(filtered_rows)}")
    emit(log_level, "INFO", f"[text-len-col] out_csv={out_csv_file}")
    return summary

