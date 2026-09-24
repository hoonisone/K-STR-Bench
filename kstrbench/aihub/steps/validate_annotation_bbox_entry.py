from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


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


def _is_bbox_none_like(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str):
        s = raw.strip().lower()
        return s in ("", "none", "null")
    return False


class ValidateAnnotationBboxStep:
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
    out_stats_csv_file_raw = params.get("out_stats_csv_file")
    bbox_col = str(params.get("bbox_col", "bbox"))
    drop_invalid_bbox_format = coerce_bool(params.get("drop_invalid_bbox_format", False), default=False)
    drop_invalid_bbox_value = coerce_bool(params.get("drop_invalid_bbox_value", False), default=False)
    check_valid_bbox_format = coerce_bool(params.get("check_valid_bbox_format", False), default=False)
    check_valid_bbox_value = coerce_bool(params.get("check_valid_bbox_value", False), default=False)
    is_valid_bbox_format_col = str(params.get("is_valid_bbox_format_col", "is_valid_bbox_format")).strip()
    is_valid_bbox_value_col = str(params.get("is_valid_bbox_value_col", "is_valid_bbox_value")).strip()
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if check_valid_bbox_format and not is_valid_bbox_format_col:
        raise ValueError("is_valid_bbox_format_col is required when check_valid_bbox_format is True")
    if check_valid_bbox_value and not is_valid_bbox_value_col:
        raise ValueError("is_valid_bbox_value_col is required when check_valid_bbox_value is True")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    out_stats_csv_file = (
        Path(str(out_stats_csv_file_raw)).expanduser().resolve()
        if out_stats_csv_file_raw
        else None
    )
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("validate_annotation_bbox_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "out_stats_csv_file": str(out_stats_csv_file) if out_stats_csv_file else None,
            "bbox_col": bbox_col,
            "drop_invalid_bbox_format": drop_invalid_bbox_format,
            "drop_invalid_bbox_value": drop_invalid_bbox_value,
            "check_valid_bbox_format": check_valid_bbox_format,
            "check_valid_bbox_value": check_valid_bbox_value,
            "is_valid_bbox_format_col": is_valid_bbox_format_col if check_valid_bbox_format else None,
            "is_valid_bbox_value_col": is_valid_bbox_value_col if check_valid_bbox_value else None,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="validate annotation bbox discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="validate annotation bbox", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    kept_rows: list[dict[str, str]] = []
    drop_counts = {
        "invalid_bbox_format": 0,
        "invalid_bbox_value": 0,
    }
    check_true_counts = {
        "valid_bbox_format": 0,
        "valid_bbox_value": 0,
    }
    dropped_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(row_iter):
        bbox_raw = row.get(bbox_col)
        invalid_format = _is_bbox_none_like(bbox_raw)
        bbox: tuple[float, float, float, float] | None = None
        if not invalid_format:
            bbox = _parse_bbox(bbox_raw)
            invalid_format = bbox is None

        invalid_value = False
        if bbox is not None:
            x, y, w, h = bbox
            invalid_value = (not all(math.isfinite(v) for v in (x, y, w, h))) or (x < 0 or y < 0 or w <= 0 or h <= 0)

        out_row = dict(row)
        if check_valid_bbox_format:
            is_valid_format = not invalid_format
            out_row[is_valid_bbox_format_col] = is_valid_format
            if is_valid_format:
                check_true_counts["valid_bbox_format"] += 1
        if check_valid_bbox_value:
            is_valid_value = not invalid_value
            out_row[is_valid_bbox_value_col] = is_valid_value
            if is_valid_value:
                check_true_counts["valid_bbox_value"] += 1

        if drop_invalid_bbox_format and invalid_format:
            drop_counts["invalid_bbox_format"] += 1
            dropped_rows.append(
                {"row_index": row_index, "reason": "invalid_bbox_format", "row": out_row}
            )
            continue
        if drop_invalid_bbox_value and invalid_value:
            drop_counts["invalid_bbox_value"] += 1
            dropped_rows.append(
                {"row_index": row_index, "reason": "invalid_bbox_value", "row": out_row}
            )
            continue

        kept_rows.append(out_row)

    out_fieldnames = list(fieldnames)
    if check_valid_bbox_format and is_valid_bbox_format_col not in out_fieldnames:
        out_fieldnames.append(is_valid_bbox_format_col)
    if check_valid_bbox_value and is_valid_bbox_value_col not in out_fieldnames:
        out_fieldnames.append(is_valid_bbox_value_col)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)

    dropped_total = sum(int(v) for v in drop_counts.values())
    if out_stats_csv_file is not None:
        out_stats_csv_file.parent.mkdir(parents=True, exist_ok=True)
        stats_rows = [
            {"metric": "input_row_count", "value": len(rows)},
            {"metric": "kept_row_count", "value": len(kept_rows)},
            {"metric": "dropped_row_count", "value": dropped_total},
        ] + [{"metric": f"drop_{k}", "value": v} for k, v in drop_counts.items()]
        with out_stats_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(stats_rows)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "input_row_count": len(rows),
        "kept_row_count": len(kept_rows),
        "dropped_row_count": dropped_total,
        "drop_counts": drop_counts,
        "check_true_counts": check_true_counts,
        "out_stats_csv_file": str(out_stats_csv_file) if out_stats_csv_file else None,
        "bbox_col": bbox_col,
        "drop_invalid_bbox_format": drop_invalid_bbox_format,
        "drop_invalid_bbox_value": drop_invalid_bbox_value,
        "check_valid_bbox_format": check_valid_bbox_format,
        "check_valid_bbox_value": check_valid_bbox_value,
        "is_valid_bbox_format_col": is_valid_bbox_format_col if check_valid_bbox_format else None,
        "is_valid_bbox_value_col": is_valid_bbox_value_col if check_valid_bbox_value else None,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    if dropped_rows:
        logger.log(
            {
                "dropped_row_count": dropped_total,
                "dropped_rows": dropped_rows,
            },
            title="validate annotation bbox dropped rows",
        )
    logger.log(summary, title="validate annotation bbox summary")
    emit(
        log_level,
        "INFO",
        (
            f"[bbox-validate] input={len(rows)}, kept={len(kept_rows)}, dropped={dropped_total}, "
            f"drop_counts={drop_counts}"
        ),
    )
    emit(log_level, "INFO", f"[bbox-validate] out_csv={out_csv_file}")
    return summary
