from __future__ import annotations

import csv
import json
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
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x = float(value[0])
            y = float(value[1])
            w = float(value[2])
            h = float(value[3])
            return x, y, w, h
        except (TypeError, ValueError):
            return None
    return None


def _is_bbox_none_like(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str):
        s = raw.strip().lower()
        return s in ("", "none", "null")
    return False


class FilterAnnotationOutliersStep:
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
    image_id_col = str(params.get("image_id_col", "image_id"))
    text_col = str(params.get("text_col", "text"))
    bbox_col = str(params.get("bbox_col", "bbox"))
    drop_empty_text = coerce_bool(params.get("drop_empty_text", True), default=True)
    drop_invalid_bbox = coerce_bool(params.get("drop_invalid_bbox", True), default=True)
    drop_nonpositive_bbox = coerce_bool(params.get("drop_nonpositive_bbox", True), default=True)
    drop_missing_id = coerce_bool(params.get("drop_missing_id", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    out_stats_csv_file = (
        Path(str(out_stats_csv_file_raw)).expanduser().resolve()
        if out_stats_csv_file_raw
        else None
    )
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("filter_annotation_outliers_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "out_stats_csv_file": str(out_stats_csv_file) if out_stats_csv_file else None,
            "image_id_col": image_id_col,
            "text_col": text_col,
            "bbox_col": bbox_col,
            "drop_empty_text": drop_empty_text,
            "drop_invalid_bbox": drop_invalid_bbox,
            "drop_nonpositive_bbox": drop_nonpositive_bbox,
            "drop_missing_id": drop_missing_id,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="filter annotation outliers discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="filter annotation outliers", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    kept_rows: list[dict[str, str]] = []
    drop_counts = {
        "missing_image_id": 0,
        "empty_text": 0,
        "bbox_none": 0,
        "invalid_bbox": 0,
        "nonpositive_bbox": 0,
    }
    dropped_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(row_iter):
        img_id = str(row.get(image_id_col, "") or "").strip()
        text = str(row.get(text_col, "") or "")
        bbox_raw = row.get(bbox_col)

        if drop_missing_id and not img_id:
            drop_counts["missing_image_id"] += 1
            dropped_rows.append(
                {"row_index": row_index, "reason": "missing_image_id", "row": row}
            )
            continue
        if drop_empty_text and not text.strip():
            drop_counts["empty_text"] += 1
            dropped_rows.append(
                {"row_index": row_index, "reason": "empty_text", "row": row}
            )
            continue
        if drop_invalid_bbox:
            if _is_bbox_none_like(bbox_raw):
                drop_counts["bbox_none"] += 1
                dropped_rows.append(
                    {"row_index": row_index, "reason": "bbox_none", "row": row}
                )
                continue
            bbox = _parse_bbox(bbox_raw)
            if bbox is None:
                drop_counts["invalid_bbox"] += 1
                dropped_rows.append(
                    {"row_index": row_index, "reason": "invalid_bbox", "row": row}
                )
                continue
            if drop_nonpositive_bbox:
                _, _, w, h = bbox
                if w <= 0 or h <= 0:
                    drop_counts["nonpositive_bbox"] += 1
                    dropped_rows.append(
                        {"row_index": row_index, "reason": "nonpositive_bbox", "row": row}
                    )
                    continue

        kept_rows.append(row)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
        "out_stats_csv_file": str(out_stats_csv_file) if out_stats_csv_file else None,
        "image_id_col": image_id_col,
        "text_col": text_col,
        "bbox_col": bbox_col,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    if dropped_rows:
        logger.log(
            {
                "dropped_row_count": dropped_total,
                "dropped_rows": dropped_rows,
            },
            title="filter annotation outliers dropped rows",
        )
    logger.log(summary, title="filter annotation outliers summary")
    emit(
        log_level,
        "INFO",
        (
            f"[ann-outlier] input={len(rows)}, kept={len(kept_rows)}, dropped={dropped_total}, "
            f"drop_counts={drop_counts}"
        ),
    )
    emit(log_level, "INFO", f"[ann-outlier] out_csv={out_csv_file}")
    return summary

