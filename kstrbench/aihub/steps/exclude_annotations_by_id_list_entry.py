from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class ExcludeAnnotationsByIdListStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file"))
    error_csv_file_raw = params.get("error_csv_file", params.get("out_error_csv_file"))
    exclude_id_list_file_raw = params.get("exclude_id_list_file")
    annotation_id_col = str(params.get("annotation_id_col", "annotation_id"))
    is_valid_crop_col = str(params.get("is_valid_crop_col", "is_valid_crop")).strip()
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not error_csv_file_raw and not exclude_id_list_file_raw:
        raise ValueError("error_csv_file (or exclude_id_list_file) is required")
    if not is_valid_crop_col:
        raise ValueError("is_valid_crop_col is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    source_id_file = Path(
        str(error_csv_file_raw if error_csv_file_raw else exclude_id_list_file_raw)
    ).expanduser().resolve()

    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")
    if not source_id_file.is_file():
        raise FileNotFoundError(f"id source file not found: {source_id_file}")

    exclude_ids: set[str] = set()
    id_source_type = "error_csv_file" if error_csv_file_raw else "exclude_id_list_file"
    if error_csv_file_raw:
        with source_id_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if fieldnames:
                id_col = annotation_id_col if annotation_id_col in fieldnames else "annotation_id"
                if id_col not in fieldnames:
                    raise ValueError(
                        f"error_csv_file must include '{annotation_id_col}' or 'annotation_id': {source_id_file}"
                    )
                for row in reader:
                    ann_id = str(row.get(id_col, "") or "").strip()
                    if ann_id:
                        exclude_ids.add(ann_id)
            else:
                f.seek(0)
                exclude_ids = {line.strip() for line in f if line.strip()}
    else:
        with source_id_file.open("r", encoding="utf-8") as f:
            exclude_ids = {line.strip() for line in f if line.strip()}

    logger = StepLogger("exclude_annotations_by_id_list_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "id_source_file": str(source_id_file),
            "id_source_type": id_source_type,
            "annotation_id_col": annotation_id_col,
            "is_valid_crop_col": is_valid_crop_col,
            "exclude_id_count": len(exclude_ids),
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="mark valid crop by id list discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        if annotation_id_col not in fieldnames:
            raise ValueError(f"annotation id column not found: {annotation_id_col}")
        rows = [dict(r) for r in reader]

    output_rows: list[dict[str, str]] = []
    marked_false_rows: list[dict[str, str]] = []

    row_iter: Any = (
        tqdm(rows, desc="mark valid crop by ids", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )
    for row in row_iter:
        out = dict(row)
        ann_id = str(row.get(annotation_id_col, "") or "").strip()
        if ann_id and ann_id in exclude_ids:
            out[is_valid_crop_col] = "False"
            marked_false_rows.append(out)
        else:
            out[is_valid_crop_col] = "True"
        output_rows.append(out)

    out_fields = list(fieldnames)
    if is_valid_crop_col not in out_fields:
        out_fields.append(is_valid_crop_col)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    marked_false_samples = marked_false_rows[:20]
    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "id_source_file": str(source_id_file),
        "id_source_type": id_source_type,
        "annotation_id_col": annotation_id_col,
        "is_valid_crop_col": is_valid_crop_col,
        "input_row_count": len(rows),
        "output_row_count": len(output_rows),
        "marked_false_row_count": len(marked_false_rows),
        "marked_true_row_count": len(output_rows) - len(marked_false_rows),
        "exclude_id_count": len(exclude_ids),
        "marked_false_sample_count": len(marked_false_samples),
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(
        {
            **summary,
            "marked_false_samples": marked_false_samples,
        },
        title="mark valid crop by id list summary",
    )
    emit(
        log_level,
        "INFO",
        (
            f"[mark-valid-crop] input={len(rows)}, false={len(marked_false_rows)}, "
            f"true={len(output_rows) - len(marked_false_rows)}, id_list={len(exclude_ids)}"
        ),
    )
    emit(log_level, "INFO", f"[mark-valid-crop] out_csv={out_csv_file}")
    return summary

