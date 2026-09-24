from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_path_text(text: str) -> str:
    return str(text).strip().replace("\\", "/")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    for part in reversed(str(value).replace("-", "_").split("_")):
        if part.isdigit():
            return int(part)
    return default


def _pick_existing_column(fieldnames: list[str], preferred: str, candidates: list[str]) -> str:
    if preferred in fieldnames:
        return preferred
    for col in candidates:
        if col in fieldnames:
            return col
    return preferred


class AssignAnnotationImagePathsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    annotation_csv_file_raw = params.get("annotation_csv_file")
    out_annotation_csv_file_raw = params.get("out_annotation_csv_file", annotation_csv_file_raw)
    source_image_id_col = str(params.get("source_image_id_col", "image_id"))
    annotation_id_col = str(params.get("annotation_id_col", "annotation_id"))
    out_image_path_col = str(params.get("out_image_path_col", "image_path"))
    text_image_root = _normalize_path_text(str(params.get("text_image_root", "")).strip("/"))
    bin_size = max(1, int(params.get("bin_size", 1000)))
    text_img_ext = str(params.get("text_img_ext", "jpg")).strip().lower().lstrip(".") or "jpg"
    overwrite_existing = coerce_bool(params.get("overwrite_existing", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not annotation_csv_file_raw:
        raise ValueError("annotation_csv_file is required")
    if not out_annotation_csv_file_raw:
        raise ValueError("out_annotation_csv_file is required")
    if not out_image_path_col:
        raise ValueError("out_image_path_col is required")

    annotation_csv_file = Path(str(annotation_csv_file_raw)).expanduser().resolve()
    out_annotation_csv_file = Path(str(out_annotation_csv_file_raw)).expanduser().resolve()
    if not annotation_csv_file.is_file():
        raise FileNotFoundError(f"annotation_csv_file not found: {annotation_csv_file}")

    logger = StepLogger("assign_annotation_image_paths_entry")
    logger.log(
        {
            "annotation_csv_file": str(annotation_csv_file),
            "out_annotation_csv_file": str(out_annotation_csv_file),
            "source_image_id_col": source_image_id_col,
            "annotation_id_col": annotation_id_col,
            "out_image_path_col": out_image_path_col,
            "text_image_root": text_image_root,
            "bin_size": bin_size,
            "text_img_ext": text_img_ext,
            "overwrite_existing": overwrite_existing,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="assign annotation image paths discovery",
    )

    with annotation_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {annotation_csv_file}")

        source_image_id_col = _pick_existing_column(
            fieldnames, source_image_id_col, ["scene_id", "source_image_id", "image_id"]
        )
        annotation_id_col = _pick_existing_column(
            fieldnames, annotation_id_col, ["text_id", "annotation_id", "id"]
        )
        if source_image_id_col not in fieldnames:
            raise ValueError(
                f"annotation_csv must include '{source_image_id_col}' (or source_image_id/image_id): {annotation_csv_file}"
            )
        if annotation_id_col not in fieldnames:
            raise ValueError(
                f"annotation_csv must include '{annotation_id_col}' (or annotation_id/id): {annotation_csv_file}"
            )
        rows = [dict(r) for r in reader]

    out_fieldnames = list(fieldnames)
    if out_image_path_col not in out_fieldnames:
        out_fieldnames.append(out_image_path_col)

    assigned_count = 0
    skipped_existing_count = 0
    skipped_missing_key_count = 0

    row_iter: Any = (
        tqdm(rows, desc="assign image_path from ids", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )
    out_rows: list[dict[str, str]] = []
    for row in row_iter:
        out = dict(row)
        existing = str(out.get(out_image_path_col, "") or "").strip()
        if existing and (not overwrite_existing):
            skipped_existing_count += 1
            out_rows.append(out)
            continue

        sid = str(out.get(source_image_id_col, "") or "").strip()
        ann_id = str(out.get(annotation_id_col, "") or "").strip()
        if not sid or not ann_id:
            skipped_missing_key_count += 1
            out_rows.append(out)
            continue

        bucket = _safe_int(sid, default=0) // max(1, bin_size)
        rel_path = _normalize_path_text(
            f"{text_image_root}/{bucket}/{ann_id}.{text_img_ext}"
            if text_image_root
            else f"{bucket}/{ann_id}.{text_img_ext}"
        )
        out[out_image_path_col] = rel_path
        assigned_count += 1
        out_rows.append(out)

    out_annotation_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_annotation_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    result = {
        "ok": True,
        "annotation_csv_file": str(annotation_csv_file),
        "out_annotation_csv_file": str(out_annotation_csv_file),
        "source_image_id_col": source_image_id_col,
        "annotation_id_col": annotation_id_col,
        "out_image_path_col": out_image_path_col,
        "text_image_root": text_image_root,
        "bin_size": bin_size,
        "text_img_ext": text_img_ext,
        "overwrite_existing": overwrite_existing,
        "show_progress": show_progress,
        "log_level": log_level,
        "row_count": len(rows),
        "assigned_count": assigned_count,
        "skipped_existing_count": skipped_existing_count,
        "skipped_missing_key_count": skipped_missing_key_count,
    }
    logger.log(result, title="assign annotation image paths summary")
    emit(
        log_level,
        "INFO",
        (
            f"[assign-ann-path] rows={len(rows)}, assigned={assigned_count}, "
            f"skipped_existing={skipped_existing_count}, skipped_missing_key={skipped_missing_key_count}"
        ),
    )
    emit(log_level, "INFO", f"[assign-ann-path] out_annotations={out_annotation_csv_file}")
    return result

