from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AnalyzeColumnVarianceStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


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


def _parse_bin_edges(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        values = [float(x.strip()) for x in s.split(",") if x.strip()]
    else:
        values = [float(raw)]
    normalized: list[float] = []
    for v in values:
        # Negative edges mean "inverse magnitude" shorthand, e.g. -2 -> 0.5.
        if v < 0:
            normalized.append(1.0 / abs(v))
        else:
            normalized.append(v)
    values = sorted(set(normalized))
    if len(values) < 2:
        return []
    return values


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
    allowed_ops = {
        "eq",
        "ne",
        "in",
        "not_in",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "not_contains",
        "regex",
        "not_regex",
    }
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"filters[{i}] must be an object")
        column = str(item.get("column", "")).strip()
        op_raw = str(item.get("op", "")).strip().lower()
        legacy_values = _as_str_list(item.get("values"))
        legacy_exclude = coerce_bool(item.get("exclude", False), default=False)
        op = op_raw or ("not_in" if legacy_exclude else "in")
        if op not in allowed_ops:
            raise ValueError(f"filters[{i}].op is invalid: {op}")

        value = item.get("value")
        values = _as_str_list(item.get("values"))
        exclude = coerce_bool(item.get("exclude", False), default=False)
        if not column:
            raise ValueError(f"filters[{i}].column is required")
        if op in ("in", "not_in"):
            if not values:
                values = legacy_values
            if not values:
                raise ValueError(f"filters[{i}].values is required for op={op}")
        elif op in ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains", "regex", "not_regex"):
            if value is None:
                raise ValueError(f"filters[{i}].value is required for op={op}")
        else:
            raise ValueError(f"unsupported filter op: {op}")

        regex = None
        if op in ("regex", "not_regex"):
            try:
                regex = re.compile(str(value))
            except re.error as exc:
                raise ValueError(f"filters[{i}].value invalid regex: {value!r}") from exc

        parsed.append(
            {
                "column": column,
                "op": op,
                "value": value,
                "values": values,
                "exclude": exclude,
                "regex": regex,
            }
        )
    return parsed


def _variance_population(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _variance_sample(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


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
    if op == "regex":
        rgx = filt.get("regex")
        return bool(rgx.search(text)) if rgx is not None else False
    if op == "not_regex":
        rgx = filt.get("regex")
        return not bool(rgx.search(text)) if rgx is not None else True

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
    target_column = str(params.get("target_column", "")).strip()
    out_stats_csv_file_raw = params.get("out_stats_csv_file")
    out_detail_csv_file_raw = params.get("out_detail_csv_file")
    is_continuous = coerce_bool(params.get("is_continuous", True), default=True)
    bin_edges = _parse_bin_edges(params.get("bin_edges"))
    filter_column = str(params.get("filter_column", "")).strip()
    filter_values = _as_str_list(params.get("filter_values"))
    filter_exclude = coerce_bool(params.get("filter_exclude", False), default=False)
    filters = _parse_filters(params.get("filters"))
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not target_column:
        raise ValueError("target_column is required")
    if not out_stats_csv_file_raw:
        raise ValueError("out_stats_csv_file is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_stats_csv_file = Path(str(out_stats_csv_file_raw)).expanduser().resolve()
    out_detail_csv_file = (
        Path(str(out_detail_csv_file_raw)).expanduser().resolve()
        if out_detail_csv_file_raw
        else out_stats_csv_file.with_name(f"{out_stats_csv_file.stem}_detail{out_stats_csv_file.suffix}")
    )

    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("analyze_column_variance_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "target_column": target_column,
            "out_stats_csv_file": str(out_stats_csv_file),
            "out_detail_csv_file": str(out_detail_csv_file),
            "is_continuous": is_continuous,
            "bin_edges": bin_edges,
            "filter_column": filter_column or None,
            "filter_values": filter_values,
            "filter_exclude": filter_exclude,
            "filters": filters,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="analyze column variance discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if target_column not in fieldnames:
            raise ValueError(f"CSV missing target_column '{target_column}': {in_csv_file}")
        if filter_column and filter_column not in fieldnames:
            raise ValueError(f"CSV missing filter_column '{filter_column}': {in_csv_file}")
        for f in filters:
            if str(f["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{f['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    input_row_count = len(rows)
    filtered_rows = rows
    if filter_column:
        if not filter_values:
            raise ValueError("filter_values is required when filter_column is provided")
        filter_set = {str(v) for v in filter_values}
        if filter_exclude:
            filtered_rows = [r for r in rows if str(r.get(filter_column, "")) not in filter_set]
        else:
            filtered_rows = [r for r in rows if str(r.get(filter_column, "")) in filter_set]
    if filters:
        for f in filters:
            f_col = str(f["column"])
            filtered_rows = [
                r
                for r in filtered_rows
                if (
                    (not bool(f.get("exclude", False)) and _evaluate_filter(r.get(f_col, ""), f))
                    or (bool(f.get("exclude", False)) and (not _evaluate_filter(r.get(f_col, ""), f)))
                )
            ]

    row_iter: Any = (
        tqdm(filtered_rows, desc="analyze column variance", unit="row", dynamic_ncols=True)
        if show_progress
        else filtered_rows
    )

    out_stats_csv_file.parent.mkdir(parents=True, exist_ok=True)
    out_detail_csv_file.parent.mkdir(parents=True, exist_ok=True)

    if is_continuous:
        numeric_values: list[float] = []
        for row_index, row in enumerate(row_iter):
            raw = row.get(target_column)
            val = _to_float(raw)
            if val is None:
                raise ValueError(
                    f"non-numeric value for continuous column at row_index={row_index}: "
                    f"{target_column}={raw!r}"
                )
            numeric_values.append(val)

        value_count = len(numeric_values)
        min_value = min(numeric_values) if numeric_values else 0.0
        max_value = max(numeric_values) if numeric_values else 0.0
        mean_value = (sum(numeric_values) / value_count) if value_count else 0.0
        var_pop = _variance_population(numeric_values)
        var_sample = _variance_sample(numeric_values)

        with out_stats_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(
                [
                    {"metric": "target_column", "value": target_column},
                    {"metric": "is_continuous", "value": "True"},
                    {"metric": "input_row_count", "value": input_row_count},
                    {"metric": "value_count", "value": value_count},
                    {"metric": "min", "value": min_value},
                    {"metric": "max", "value": max_value},
                    {"metric": "mean", "value": mean_value},
                    {"metric": "variance_population", "value": var_pop},
                    {"metric": "variance_sample", "value": var_sample},
                ]
            )

        if bin_edges:
            bins: list[dict[str, Any]] = []
            for i in range(len(bin_edges) - 1):
                left = bin_edges[i]
                right = bin_edges[i + 1]
                bins.append(
                    {
                        "bin_index": i,
                        "bin_left": left,
                        "bin_right": right,
                        "count": 0,
                    }
                )
            below_min = 0
            above_max = 0
            for v in numeric_values:
                if v < bin_edges[0]:
                    below_min += 1
                    continue
                if v > bin_edges[-1]:
                    above_max += 1
                    continue
                placed = False
                for i in range(len(bin_edges) - 1):
                    left = bin_edges[i]
                    right = bin_edges[i + 1]
                    is_last = i == len(bin_edges) - 2
                    if (left <= v < right) or (is_last and left <= v <= right):
                        bins[i]["count"] += 1
                        placed = True
                        break
                if not placed:
                    above_max += 1

            with out_detail_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["bin_index", "bin_left", "bin_right", "count"],
                )
                writer.writeheader()
                writer.writerows(bins)
                writer.writerow(
                    {
                        "bin_index": "below_min",
                        "bin_left": "",
                        "bin_right": bin_edges[0],
                        "count": below_min,
                    }
                )
                writer.writerow(
                    {
                        "bin_index": "above_max",
                        "bin_left": bin_edges[-1],
                        "bin_right": "",
                        "count": above_max,
                    }
                )
        else:
            with out_detail_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["value"])
                writer.writeheader()
                for v in numeric_values:
                    writer.writerow({"value": v})

        summary = {
            "ok": True,
            "in_csv_file": str(in_csv_file),
            "target_column": target_column,
            "is_continuous": True,
            "input_row_count": input_row_count,
            "value_count": value_count,
            "variance_population": var_pop,
            "variance_sample": var_sample,
            "out_stats_csv_file": str(out_stats_csv_file),
            "out_detail_csv_file": str(out_detail_csv_file),
            "bin_edges": bin_edges,
            "filter_column": filter_column or None,
            "filter_values": filter_values,
            "filter_exclude": filter_exclude,
            "filters": filters,
            "show_progress": show_progress,
            "log_level": log_level,
        }
    else:
        counts: dict[str, int] = {}
        for row in row_iter:
            raw = row.get(target_column)
            key = str(raw) if raw is not None else ""
            counts[key] = counts.get(key, 0) + 1

        freq_values = [float(v) for v in counts.values()]
        var_pop = _variance_population(freq_values)
        var_sample = _variance_sample(freq_values)

        with out_stats_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(
                [
                    {"metric": "target_column", "value": target_column},
                    {"metric": "is_continuous", "value": "False"},
                    {"metric": "input_row_count", "value": input_row_count},
                    {"metric": "row_count", "value": len(filtered_rows)},
                    {"metric": "unique_value_count", "value": len(counts)},
                    {"metric": "frequency_variance_population", "value": var_pop},
                    {"metric": "frequency_variance_sample", "value": var_sample},
                ]
            )

        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        with out_detail_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["value", "count"])
            writer.writeheader()
            for value, count in sorted_items:
                writer.writerow({"value": value, "count": count})

        summary = {
            "ok": True,
            "in_csv_file": str(in_csv_file),
            "target_column": target_column,
            "is_continuous": False,
            "input_row_count": input_row_count,
            "row_count": len(filtered_rows),
            "unique_value_count": len(counts),
            "frequency_variance_population": var_pop,
            "frequency_variance_sample": var_sample,
            "out_stats_csv_file": str(out_stats_csv_file),
            "out_detail_csv_file": str(out_detail_csv_file),
            "filter_column": filter_column or None,
            "filter_values": filter_values,
            "filter_exclude": filter_exclude,
            "filters": filters,
            "show_progress": show_progress,
            "log_level": log_level,
        }

    logger.log(summary, title="analyze column variance summary")
    emit(log_level, "INFO", f"[col-var] target={target_column}, continuous={is_continuous}")
    emit(log_level, "INFO", f"[col-var] out_stats={out_stats_csv_file}")
    emit(log_level, "INFO", f"[col-var] out_detail={out_detail_csv_file}")
    return summary

