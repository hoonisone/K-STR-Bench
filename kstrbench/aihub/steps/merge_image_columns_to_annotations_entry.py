from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _pick_existing_column(fieldnames: list[str], preferred: str, candidates: list[str]) -> str:
    if preferred in fieldnames:
        return preferred
    for name in candidates:
        if name in fieldnames:
            return name
    return preferred


class MergeImageColumnsToAnnotationsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    image_csv_file_raw = params.get("image_csv_file")
    annotation_csv_file_raw = params.get("annotation_csv_file")
    out_annotation_csv_file_raw = params.get("out_annotation_csv_file", annotation_csv_file_raw)
    image_id_col = str(params.get("image_id_col", "image_id"))
    annotation_image_id_col = str(params.get("annotation_image_id_col", "source_image_id"))
    merge_columns = _as_str_list(params.get("merge_columns"))
    overwrite_existing = coerce_bool(params.get("overwrite_existing", True), default=True)
    strict_columns = coerce_bool(params.get("strict_columns", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not image_csv_file_raw:
        raise ValueError("image_csv_file is required")
    if not annotation_csv_file_raw:
        raise ValueError("annotation_csv_file is required")
    if not out_annotation_csv_file_raw:
        raise ValueError("out_annotation_csv_file is required")
    if not merge_columns:
        raise ValueError("merge_columns is required")

    image_csv_file = Path(str(image_csv_file_raw)).expanduser().resolve()
    annotation_csv_file = Path(str(annotation_csv_file_raw)).expanduser().resolve()
    out_annotation_csv_file = Path(str(out_annotation_csv_file_raw)).expanduser().resolve()

    if not image_csv_file.is_file():
        raise FileNotFoundError(f"image_csv_file not found: {image_csv_file}")
    if not annotation_csv_file.is_file():
        raise FileNotFoundError(f"annotation_csv_file not found: {annotation_csv_file}")

    logger = StepLogger("merge_image_columns_to_annotations_entry")

    with image_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        image_reader = csv.DictReader(f)
        image_fieldnames = list(image_reader.fieldnames or [])
        image_id_col = _pick_existing_column(image_fieldnames, image_id_col, ["image_id", "source_image_id"])
        if image_id_col not in image_fieldnames:
            raise ValueError(f"image_csv must include '{image_id_col}': {image_csv_file}")

        missing_merge_columns = [c for c in merge_columns if c not in image_fieldnames]
        if strict_columns and missing_merge_columns:
            raise ValueError(f"missing merge columns in image_csv: {missing_merge_columns}")

        image_rows = [dict(r) for r in image_reader]

    image_by_id: dict[str, dict[str, str]] = {}
    for row in image_rows:
        key = str(row.get(image_id_col, "") or "").strip()
        if key and key not in image_by_id:
            image_by_id[key] = row

    with annotation_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        ann_reader = csv.DictReader(f)
        ann_fieldnames = list(ann_reader.fieldnames or [])
        annotation_image_id_col = _pick_existing_column(
            ann_fieldnames, annotation_image_id_col, ["source_image_id", "image_id"]
        )
        if annotation_image_id_col not in ann_fieldnames:
            raise ValueError(
                f"annotation_csv must include '{annotation_image_id_col}': {annotation_csv_file}"
            )
        annotation_rows = [dict(r) for r in ann_reader]

    out_fieldnames = list(ann_fieldnames)
    for col in merge_columns:
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    unresolved_annotation_count = 0
    merged_cell_count = 0
    skipped_cell_count = 0
    missing_source_column_count = 0

    ann_iter: Any = (
        tqdm(annotation_rows, desc="merge image cols -> annotations", unit="row", dynamic_ncols=True)
        if show_progress
        else annotation_rows
    )
    out_rows: list[dict[str, str]] = []
    for ann in ann_iter:
        out = dict(ann)
        ann_id = str(ann.get(annotation_image_id_col, "") or "").strip()
        src = image_by_id.get(ann_id)
        if src is None:
            unresolved_annotation_count += 1
            out_rows.append(out)
            continue

        for col in merge_columns:
            if col not in src:
                missing_source_column_count += 1
                if strict_columns:
                    raise ValueError(f"source column not found in image row: {col}")
                continue

            target_existing = str(out.get(col, "") or "").strip()
            if (not overwrite_existing) and target_existing:
                skipped_cell_count += 1
                continue

            out[col] = src.get(col, "")
            merged_cell_count += 1

        out_rows.append(out)

    out_annotation_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_annotation_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    result = {
        "ok": True,
        "image_csv_file": str(image_csv_file),
        "annotation_csv_file": str(annotation_csv_file),
        "out_annotation_csv_file": str(out_annotation_csv_file),
        "image_id_col": image_id_col,
        "annotation_image_id_col": annotation_image_id_col,
        "merge_columns": merge_columns,
        "overwrite_existing": overwrite_existing,
        "strict_columns": strict_columns,
        "show_progress": show_progress,
        "log_level": log_level,
        "image_row_count": len(image_rows),
        "image_key_count": len(image_by_id),
        "annotation_row_count": len(annotation_rows),
        "unresolved_annotation_count": unresolved_annotation_count,
        "merged_cell_count": merged_cell_count,
        "skipped_cell_count": skipped_cell_count,
        "missing_source_column_count": missing_source_column_count,
    }
    logger.log(result, title="merge image columns summary")
    emit(
        log_level,
        "INFO",
        (
            f"[merge-cols] ann={len(annotation_rows)}, merged_cells={merged_cell_count}, "
            f"unresolved={unresolved_annotation_count}, skipped_cells={skipped_cell_count}"
        ),
    )
    emit(log_level, "INFO", f"[merge-cols] out={out_annotation_csv_file}")
    return result

