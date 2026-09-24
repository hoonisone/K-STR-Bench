from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class SampleMergeCsvsByRatioStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _as_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _alloc_counts(total: int, weights: list[float]) -> list[int]:
    if total < 0:
        raise ValueError("total_sample_count must be >= 0")
    s = sum(weights)
    if s <= 0:
        raise ValueError("sum(sample_ratio_list) must be > 0")
    normalized = [w / s for w in weights]
    raw = [total * w for w in normalized]
    base = [int(x) for x in raw]
    remain = total - sum(base)
    frac_idx = sorted(
        [(raw[i] - base[i], i) for i in range(len(weights))],
        key=lambda x: (-x[0], x[1]),
    )
    for i in range(remain):
        base[frac_idx[i][1]] += 1
    return base


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
    csv_file_list_raw = params.get("csv_file_list")
    sample_ratio_list_raw = params.get("sample_ratio_list")
    total_sample_count = int(params.get("total_sample_count", 0))
    out_csv_file_raw = params.get("out_csv_file")
    filters = _parse_filters(params.get("filters"))
    seed = int(params.get("seed", 42))
    use_shuffle = coerce_bool(params.get("use_shuffle", True), default=True)
    sample_with_replacement = coerce_bool(params.get("sample_with_replacement", False), default=False)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    csv_file_list = [str(x).strip() for x in _as_list(csv_file_list_raw) if str(x).strip()]
    sample_ratio_list = [float(x) for x in _as_list(sample_ratio_list_raw)]

    if not csv_file_list:
        raise ValueError("csv_file_list is required")
    if not sample_ratio_list:
        raise ValueError("sample_ratio_list is required")
    if len(csv_file_list) != len(sample_ratio_list):
        raise ValueError("csv_file_list and sample_ratio_list lengths must match")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file is required")

    csv_files = [Path(p).expanduser().resolve() for p in csv_file_list]
    for p in csv_files:
        if not p.is_file():
            raise FileNotFoundError(f"csv file not found: {p}")
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()

    target_counts = _alloc_counts(total_sample_count, sample_ratio_list)
    rng = random.Random(seed)

    all_rows: list[dict[str, str]] = []
    merged_fieldnames: list[str] = []
    sampled_counts: list[int] = []
    source_row_counts: list[int] = []
    filtered_row_counts: list[int] = []

    for csv_path, target_count in zip(csv_files, target_counts):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")
            for flt in filters:
                if str(flt["column"]) not in fieldnames:
                    raise ValueError(f"CSV missing filters.column '{flt['column']}': {csv_path}")
            for col in fieldnames:
                if col not in merged_fieldnames:
                    merged_fieldnames.append(col)
            rows = [dict(r) for r in reader]

        n = len(rows)
        source_row_counts.append(n)
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
        rows = filtered_rows
        filtered_row_counts.append(len(rows))
        if target_count == 0:
            sampled_counts.append(0)
            continue

        if sample_with_replacement:
            picked = [rng.choice(rows) for _ in range(target_count)] if n > 0 else []
        else:
            if target_count > n:
                raise ValueError(
                    f"requested sample_count ({target_count}) exceeds rows ({n}) in {csv_path}"
                )
            picked = rng.sample(rows, target_count)
        sampled_counts.append(len(picked))
        all_rows.extend(picked)

    if use_shuffle:
        rng.shuffle(all_rows)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    logger = StepLogger("sample_merge_csvs_by_ratio_entry")
    summary = {
        "ok": True,
        "csv_file_list": [str(p) for p in csv_files],
        "source_row_counts": source_row_counts,
        "filtered_row_counts": filtered_row_counts,
        "filters": filters,
        "sample_ratio_list": sample_ratio_list,
        "total_sample_count": total_sample_count,
        "target_counts": target_counts,
        "sampled_counts": sampled_counts,
        "output_row_count": len(all_rows),
        "out_csv_file": str(out_csv_file),
        "seed": seed,
        "use_shuffle": use_shuffle,
        "sample_with_replacement": sample_with_replacement,
        "log_level": log_level,
    }
    logger.log(summary, title="sample merge csvs by ratio summary")
    emit(log_level, "INFO", f"[sample-merge] out_rows={len(all_rows)} -> {out_csv_file}")
    return summary

