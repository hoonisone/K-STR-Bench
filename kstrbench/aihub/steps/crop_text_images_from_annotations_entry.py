from __future__ import annotations

import csv
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_path_text(text: str) -> str:
    return text.strip().replace("\\", "/")


def _resolve_abs_path(base_dir: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    return p.expanduser().resolve() if p.is_absolute() else (base_dir / p).resolve()


def _parse_parallelism(params: dict[str, Any]) -> tuple[int, int]:
    p_raw = params.get("num_process", params.get("num_proc", 1))
    t_raw = params.get("num_thread", params.get("num_threads", params.get("num_thr", 1)))
    num_process = max(1, int(p_raw))
    num_thread = max(1, int(t_raw))
    return num_process, num_thread


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except Exception:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        if all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in raw[:4]):
            xs = [float(v[0]) for v in raw[:4]]
            ys = [float(v[1]) for v in raw[:4]]
            left = min(xs)
            top = min(ys)
            return left, top, max(xs) - left, max(ys) - top
        x = float(raw[0])
        y = float(raw[1])
        w = float(raw[2])
        h = float(raw[3])
        return x, y, w, h
    except (TypeError, ValueError):
        return None


def _chunk_rows(rows: list[dict[str, str]], n: int) -> list[list[dict[str, str]]]:
    if n <= 1 or len(rows) <= 1:
        return [rows]
    chunks: list[list[dict[str, str]]] = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        chunks[i % n].append(row)
    return [c for c in chunks if c]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    for part in reversed(str(value).replace("-", "_").split("_")):
        if part.isdigit():
            return int(part)
    return default


def _pick_existing_column(
    fieldnames: list[str],
    preferred: str,
    candidates: list[str],
) -> str:
    if preferred in fieldnames:
        return preferred
    for col in candidates:
        if col in fieldnames:
            return col
    return preferred


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


def _process_one_annotation(
    row: dict[str, str],
    image_path_by_id: dict[str, str],
    dataset_dir_str: str,
    out_dataset_dir_str: str,
    source_image_id_col: str,
    annotation_id_col: str,
    bbox_col: str,
    out_image_path_col: str,
    text_image_root: str,
    bin_size: int,
    text_img_ext: str,
    text_img_qual: int,
    dry_run: bool,
    skip_missing_files: bool,
    ignore_exif_orientation: bool,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir_str)
    out_dataset_dir = Path(out_dataset_dir_str)

    source_image_id = str(row.get(source_image_id_col, "") or "").strip()
    annotation_id = str(row.get(annotation_id_col, "") or "").strip()
    if not source_image_id or not annotation_id:
        return {"status": "skipped_invalid_row", "row": row}

    bbox = _parse_bbox(row.get(bbox_col))
    if bbox is None:
        return {"status": "skipped_invalid_bbox", "row": row}

    image_rel = str(image_path_by_id.get(source_image_id, "") or "").strip()
    if not image_rel:
        if skip_missing_files:
            return {"status": "missing_image_map", "row": row}
        raise KeyError(f"no source image mapping for id={source_image_id}")

    src_image_abs = _resolve_abs_path(dataset_dir, image_rel)
    if not src_image_abs.is_file():
        if skip_missing_files:
            return {"status": "missing_source_image", "row": row}
        raise FileNotFoundError(f"source image not found: {src_image_abs}")

    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow is required for crop_text_images_from_annotations_entry") from exc

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return {"status": "skipped_invalid_bbox", "row": row}

    try:
        with Image.open(src_image_abs) as img:
            if not ignore_exif_orientation:
                # Respect EXIF orientation before bbox-based crop.
                img = ImageOps.exif_transpose(img)
            img_w, img_h = img.size
            left = max(0, min(img_w, int(round(x))))
            top = max(0, min(img_h, int(round(y))))
            right = max(left + 1, min(img_w, int(round(x + w))))
            bottom = max(top + 1, min(img_h, int(round(y + h))))
            crop_w = max(0, right - left)
            crop_h = max(0, bottom - top)

            bucket = _safe_int(source_image_id, default=0) // max(1, bin_size)
            target_rel = (
                f"{text_image_root}/{bucket}/{annotation_id}.{text_img_ext}"
                if text_image_root
                else f"{bucket}/{annotation_id}.{text_img_ext}"
            )
            target_rel = _normalize_path_text(target_rel)
            target_abs = _resolve_abs_path(out_dataset_dir, target_rel)

            if not dry_run:
                target_abs.parent.mkdir(parents=True, exist_ok=True)
                cropped = img.crop((left, top, right, bottom))
                save_kwargs: dict[str, Any] = {}
                ext_lower = text_img_ext.lower()
                if ext_lower in ("jpg", "jpeg"):
                    save_kwargs["quality"] = text_img_qual
                cropped.save(target_abs, **save_kwargs)
    except Exception as exc:
        return {
            "status": "processing_error",
            "row": row,
            "source_image_id": source_image_id,
            "annotation_id": annotation_id,
            "source_image_rel": image_rel,
            "source_image_abs": str(src_image_abs),
            "bbox": [x, y, w, h],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    out_row = dict(row)
    out_row[out_image_path_col] = target_rel
    out_row["width"] = str(crop_w)
    out_row["height"] = str(crop_h)
    return {"status": "ok", "row": out_row}


def _run_chunk(
    rows: list[dict[str, str]],
    image_path_by_id: dict[str, str],
    dataset_dir_str: str,
    out_dataset_dir_str: str,
    source_image_id_col: str,
    annotation_id_col: str,
    bbox_col: str,
    out_image_path_col: str,
    text_image_root: str,
    bin_size: int,
    text_img_ext: str,
    text_img_qual: int,
    dry_run: bool,
    skip_missing_files: bool,
    ignore_exif_orientation: bool,
    num_thread: int,
    log_level: str = "INFO",
    show_progress: bool = False,
    progress_desc: str = "crop annotations",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    if show_progress:
        # requested behavior: show progress only on the first process / first thread lane
        num_thread = 1

    if num_thread <= 1:
        row_iter: Any = (
            tqdm(rows, desc=progress_desc, unit="ann", dynamic_ncols=True)
            if show_progress
            else rows
        )
        for row in row_iter:
            results.append(
                _process_one_annotation(
                    row=row,
                    image_path_by_id=image_path_by_id,
                    dataset_dir_str=dataset_dir_str,
                    out_dataset_dir_str=out_dataset_dir_str,
                    source_image_id_col=source_image_id_col,
                    annotation_id_col=annotation_id_col,
                    bbox_col=bbox_col,
                    out_image_path_col=out_image_path_col,
                    text_image_root=text_image_root,
                    bin_size=bin_size,
                    text_img_ext=text_img_ext,
                    text_img_qual=text_img_qual,
                    dry_run=dry_run,
                    skip_missing_files=skip_missing_files,
                    ignore_exif_orientation=ignore_exif_orientation,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=num_thread) as ex:
            futs = [
                ex.submit(
                    _process_one_annotation,
                    row,
                    image_path_by_id,
                    dataset_dir_str,
                    out_dataset_dir_str,
                    source_image_id_col,
                    annotation_id_col,
                    bbox_col,
                    out_image_path_col,
                    text_image_root,
                    bin_size,
                    text_img_ext,
                    text_img_qual,
                    dry_run,
                    skip_missing_files,
                    ignore_exif_orientation,
                )
                for row in rows
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    out_rows: list[dict[str, str]] = []
    ok_count = 0
    skipped_invalid_row = 0
    skipped_invalid_bbox = 0
    missing_image_map = 0
    missing_source_image = 0
    processing_error_count = 0
    processing_errors: list[dict[str, Any]] = []
    error_samples: list[dict[str, str]] = []

    def _append_error_sample(result_item: dict[str, Any], error_type: str) -> None:
        ann_id = str(result_item.get("annotation_id", "") or "").strip()
        if not ann_id:
            row_payload = result_item.get("row")
            if isinstance(row_payload, dict):
                ann_id = str(row_payload.get(annotation_id_col, "") or "").strip()
        error_samples.append(
            {
                "annotation_id": ann_id,
                "error_type": error_type,
            }
        )

    for result in results:
        status = str(result.get("status"))
        if status == "ok":
            ok_count += 1
            out_rows.append(result["row"])
        elif status == "skipped_invalid_row":
            skipped_invalid_row += 1
            out_rows.append(result["row"])
            _append_error_sample(result, "skipped_invalid_row")
        elif status == "skipped_invalid_bbox":
            skipped_invalid_bbox += 1
            out_rows.append(result["row"])
            _append_error_sample(result, "skipped_invalid_bbox")
        elif status == "missing_image_map":
            missing_image_map += 1
            out_rows.append(result["row"])
            _append_error_sample(result, "missing_image_map")
        elif status == "missing_source_image":
            missing_source_image += 1
            out_rows.append(result["row"])
            _append_error_sample(result, "missing_source_image")
        elif status == "processing_error":
            processing_error_count += 1
            out_rows.append(result["row"])
            error_payload = {
                "source_image_id": result.get("source_image_id"),
                "annotation_id": result.get("annotation_id"),
                "source_image_rel": result.get("source_image_rel"),
                "source_image_abs": result.get("source_image_abs"),
                "bbox": result.get("bbox"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
                "traceback": result.get("traceback"),
            }
            processing_errors.append(error_payload)
            _append_error_sample(result, "processing_error")
            emit(
                log_level,
                "ERROR",
                (
                    "[crop-ann][ERROR] "
                    f"source_image_id={error_payload['source_image_id']}, "
                    f"annotation_id={error_payload['annotation_id']}, "
                    f"source_image={error_payload['source_image_abs']}, "
                    f"bbox={error_payload['bbox']}, "
                    f"{error_payload['error_type']}: {error_payload['error_message']}"
                ),
            )

    return {
        "rows": out_rows,
        "ok_count": ok_count,
        "skipped_invalid_row_count": skipped_invalid_row,
        "skipped_invalid_bbox_count": skipped_invalid_bbox,
        "missing_image_map_count": missing_image_map,
        "missing_source_image_count": missing_source_image,
        "processing_error_count": processing_error_count,
        "processing_errors": processing_errors,
        "error_samples": error_samples,
    }


class CropTextImagesFromAnnotationsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    out_dataset_dir_raw = params.get("out_dataset_dir", dataset_dir_raw)
    image_csv_file_raw = params.get("image_csv_file")
    annotation_csv_file_raw = params.get("annotation_csv_file")
    out_annotation_csv_file_raw = params.get("out_annotation_csv_file")
    out_error_csv_file_raw = params.get("out_error_csv_file")

    source_image_id_col = str(params.get("source_image_id_col", "source_image_id"))
    source_image_path_col = str(params.get("source_image_path_col", "image_file"))
    annotation_id_col = str(params.get("annotation_id_col", "annotation_id"))
    bbox_col = str(params.get("bbox_col", "bbox"))
    out_image_path_col = str(params.get("out_image_path_col", "image_path"))

    text_image_root = _normalize_path_text(str(params.get("text_image_root", "")).strip("/"))
    bin_size = max(1, int(params.get("bin_size", 1000)))
    text_img_ext = str(params.get("text_img_ext", "jpg")).strip().lower().lstrip(".") or "jpg"
    text_img_qual = int(params.get("text_img_qual", 95))
    skip_if_out_dataset_exists = coerce_bool(params.get("skip_if_out_dataset_exists", True), default=True)
    dry_run = coerce_bool(params.get("dry_run", False), default=False)
    skip_missing_files = coerce_bool(params.get("skip_missing_files", True), default=True)
    ignore_exif_orientation = coerce_bool(params.get("ignore_exif_orientation", False), default=False)
    filters = _parse_filters(params.get("filters"))
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    num_process, num_thread = _parse_parallelism(params)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not dataset_dir_raw:
        raise ValueError("dataset_dir (or datadir) is required")
    if not image_csv_file_raw:
        raise ValueError("image_csv_file is required")
    if not annotation_csv_file_raw:
        raise ValueError("annotation_csv_file is required")

    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    out_dataset_dir = Path(str(out_dataset_dir_raw)).expanduser().resolve()
    image_csv_file = Path(str(image_csv_file_raw)).expanduser().resolve()
    annotation_csv_file = Path(str(annotation_csv_file_raw)).expanduser().resolve()
    out_error_csv_file = (
        Path(str(out_error_csv_file_raw)).expanduser().resolve()
        if out_error_csv_file_raw
        else None
    )

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    if not image_csv_file.is_file():
        raise FileNotFoundError(f"image_csv_file not found: {image_csv_file}")
    if not annotation_csv_file.is_file():
        raise FileNotFoundError(f"annotation_csv_file not found: {annotation_csv_file}")

    logger = StepLogger("crop_text_images_from_annotations_entry")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "image_csv_file": str(image_csv_file),
            "annotation_csv_file": str(annotation_csv_file),
            "out_annotation_csv_file": str(out_annotation_csv_file_raw) if out_annotation_csv_file_raw else None,
            "out_error_csv_file": str(out_error_csv_file) if out_error_csv_file else None,
            "source_image_id_col": source_image_id_col,
            "source_image_path_col": source_image_path_col,
            "annotation_id_col": annotation_id_col,
            "bbox_col": bbox_col,
            "out_image_path_col": out_image_path_col,
            "text_image_root": text_image_root,
            "bin_size": bin_size,
            "text_img_ext": text_img_ext,
            "text_img_qual": text_img_qual,
            "skip_if_out_dataset_exists": skip_if_out_dataset_exists,
            "dry_run": dry_run,
            "skip_missing_files": skip_missing_files,
            "ignore_exif_orientation": ignore_exif_orientation,
            "filters": filters,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
        },
        title="crop annotations discovery",
    )

    crop_output_root = (
        (out_dataset_dir / text_image_root) if text_image_root else out_dataset_dir
    )
    if skip_if_out_dataset_exists and crop_output_root.exists():
        has_any_output = any(crop_output_root.iterdir())
        if has_any_output:
            summary = {
                "ok": True,
                "skipped": True,
                "reason": "out_dataset already exists and not empty",
                "out_dataset_dir": str(out_dataset_dir),
                "crop_output_root": str(crop_output_root),
                "skip_if_out_dataset_exists": True,
            }
            logger.log(summary, title="crop annotations skipped")
            emit(log_level, "INFO", f"[crop-ann] skip existing output root: {crop_output_root}")
            if out_error_csv_file is not None and not out_error_csv_file.is_file():
                out_error_csv_file.parent.mkdir(parents=True, exist_ok=True)
                with out_error_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["annotation_id", "error_type"])
                    writer.writeheader()
            return summary

    with image_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        image_reader = csv.DictReader(f)
        image_fieldnames = list(image_reader.fieldnames or [])
        source_image_id_col = _pick_existing_column(
            image_fieldnames,
            source_image_id_col,
            ["scene_id", "source_image_id", "image_id"],
        )
        source_image_path_col = _pick_existing_column(
            image_fieldnames,
            source_image_path_col,
            ["image_file", "image_path", "file_name"],
        )
        if source_image_id_col not in image_fieldnames or source_image_path_col not in image_fieldnames:
            raise ValueError(
                f"image_csv must include '{source_image_id_col}' and '{source_image_path_col}': {image_csv_file}"
            )
        image_rows = [dict(r) for r in image_reader]

    image_path_by_id: dict[str, str] = {}
    for row in image_rows:
        sid = str(row.get(source_image_id_col, "") or "").strip()
        pth = str(row.get(source_image_path_col, "") or "").strip()
        if sid and pth and sid not in image_path_by_id:
            image_path_by_id[sid] = _normalize_path_text(pth)

    with annotation_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        ann_reader = csv.DictReader(f)
        ann_fieldnames = list(ann_reader.fieldnames or [])
        source_image_id_col = _pick_existing_column(
            ann_fieldnames,
            source_image_id_col,
            ["scene_id", "source_image_id", "image_id"],
        )
        annotation_id_col = _pick_existing_column(
            ann_fieldnames,
            annotation_id_col,
            ["text_id", "annotation_id", "id"],
        )
        bbox_col = _pick_existing_column(
            ann_fieldnames,
            bbox_col,
            ["bbox", "box"],
        )
        if annotation_id_col not in ann_fieldnames or bbox_col not in ann_fieldnames:
            raise ValueError(
                f"annotation_csv must include '{annotation_id_col}' and '{bbox_col}': {annotation_csv_file}"
            )
        if source_image_id_col not in ann_fieldnames:
            raise ValueError(
                f"annotation_csv must include '{source_image_id_col}' for image lookup: {annotation_csv_file}"
            )
        for flt in filters:
            if str(flt["column"]) not in ann_fieldnames:
                raise ValueError(f"annotation_csv missing filters.column '{flt['column']}': {annotation_csv_file}")
        annotation_rows = [dict(r) for r in ann_reader]

    input_annotation_count = len(annotation_rows)
    if filters:
        for flt in filters:
            col = str(flt["column"])
            annotation_rows = [
                r
                for r in annotation_rows
                if (
                    (not bool(flt.get("exclude", False)) and _evaluate_filter(r.get(col, ""), flt))
                    or (bool(flt.get("exclude", False)) and (not _evaluate_filter(r.get(col, ""), flt)))
                )
            ]

    chunks = _chunk_rows(annotation_rows, num_process)
    out_rows: list[dict[str, str]] = []
    ok_count = 0
    skipped_invalid_row_count = 0
    skipped_invalid_bbox_count = 0
    missing_image_map_count = 0
    missing_source_image_count = 0
    processing_error_count = 0
    processing_errors: list[dict[str, Any]] = []
    error_samples: list[dict[str, str]] = []

    if num_process <= 1:
        result = _run_chunk(
            rows=chunks[0],
            image_path_by_id=image_path_by_id,
            dataset_dir_str=str(dataset_dir),
            out_dataset_dir_str=str(out_dataset_dir),
            source_image_id_col=source_image_id_col,
            annotation_id_col=annotation_id_col,
            bbox_col=bbox_col,
            out_image_path_col=out_image_path_col,
            text_image_root=text_image_root,
            bin_size=bin_size,
            text_img_ext=text_img_ext,
            text_img_qual=text_img_qual,
            dry_run=dry_run,
            skip_missing_files=skip_missing_files,
            ignore_exif_orientation=ignore_exif_orientation,
            num_thread=num_thread,
            log_level=log_level,
            show_progress=show_progress,
            progress_desc="crop annotations p0:t0",
        )
        out_rows.extend(result["rows"])
        ok_count += int(result["ok_count"])
        skipped_invalid_row_count += int(result["skipped_invalid_row_count"])
        skipped_invalid_bbox_count += int(result["skipped_invalid_bbox_count"])
        missing_image_map_count += int(result["missing_image_map_count"])
        missing_source_image_count += int(result["missing_source_image_count"])
        processing_error_count += int(result.get("processing_error_count", 0))
        processing_errors.extend(list(result.get("processing_errors", [])))
        error_samples.extend(list(result.get("error_samples", [])))
    else:
        with ProcessPoolExecutor(max_workers=num_process) as ex:
            futs = [
                ex.submit(
                    _run_chunk,
                    chunk,
                    image_path_by_id,
                    str(dataset_dir),
                    str(out_dataset_dir),
                    source_image_id_col,
                    annotation_id_col,
                    bbox_col,
                    out_image_path_col,
                    text_image_root,
                    bin_size,
                    text_img_ext,
                    text_img_qual,
                    dry_run,
                    skip_missing_files,
                    ignore_exif_orientation,
                    num_thread,
                    log_level,
                    show_progress and i == 0,
                    "crop annotations p0:t0",
                )
                for i, chunk in enumerate(chunks)
            ]
            for fut in as_completed(futs):
                result = fut.result()
                out_rows.extend(result["rows"])
                ok_count += int(result["ok_count"])
                skipped_invalid_row_count += int(result["skipped_invalid_row_count"])
                skipped_invalid_bbox_count += int(result["skipped_invalid_bbox_count"])
                missing_image_map_count += int(result["missing_image_map_count"])
                missing_source_image_count += int(result["missing_source_image_count"])
                processing_error_count += int(result.get("processing_error_count", 0))
                processing_errors.extend(list(result.get("processing_errors", [])))
                error_samples.extend(list(result.get("error_samples", [])))

    if out_error_csv_file is not None:
        out_error_csv_file.parent.mkdir(parents=True, exist_ok=True)
        with out_error_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["annotation_id", "error_type"])
            writer.writeheader()
            writer.writerows(error_samples)

    summary = {
        "ok": True,
        "skipped": False,
        "dataset_dir": str(dataset_dir),
        "out_dataset_dir": str(out_dataset_dir),
        "image_csv_file": str(image_csv_file),
        "annotation_csv_file": str(annotation_csv_file),
        "out_annotation_csv_file": None,
        "annotation_csv_rewrite_enabled": False,
        "source_image_id_col": source_image_id_col,
        "source_image_path_col": source_image_path_col,
        "annotation_id_col": annotation_id_col,
        "bbox_col": bbox_col,
        "out_image_path_col": out_image_path_col,
        "text_image_root": text_image_root,
        "bin_size": bin_size,
        "text_img_ext": text_img_ext,
        "text_img_qual": text_img_qual,
        "skip_if_out_dataset_exists": skip_if_out_dataset_exists,
        "dry_run": dry_run,
        "skip_missing_files": skip_missing_files,
        "ignore_exif_orientation": ignore_exif_orientation,
        "show_progress": show_progress,
        "num_process": num_process,
        "num_thread": num_thread,
        "filters": filters,
        "input_annotation_count": input_annotation_count,
        "annotation_count": len(annotation_rows),
        "cropped_count": ok_count,
        "skipped_invalid_row_count": skipped_invalid_row_count,
        "skipped_invalid_bbox_count": skipped_invalid_bbox_count,
        "missing_image_map_count": missing_image_map_count,
        "missing_source_image_count": missing_source_image_count,
        "processing_error_count": processing_error_count,
        "error_sample_count": len(error_samples),
        "out_error_csv_file": str(out_error_csv_file) if out_error_csv_file else None,
        "log_level": log_level,
    }
    if processing_errors:
        logger.log(
            {
                "processing_error_count": processing_error_count,
                "errors": processing_errors,
            },
            title="crop annotations processing errors",
        )
    logger.log(summary, title="crop annotations summary")
    emit(
        log_level,
        "INFO",
        (
            f"[crop-ann] ann={len(annotation_rows)}, cropped={ok_count}, "
            f"invalid_row={skipped_invalid_row_count}, invalid_bbox={skipped_invalid_bbox_count}, "
            f"missing_map={missing_image_map_count}, missing_src={missing_source_image_count}, "
            f"processing_error={processing_error_count}"
        ),
    )
    if out_annotation_csv_file_raw:
        emit(
            log_level,
            "INFO",
            "[crop-ann] note: out_annotation_csv_file is ignored (annotation CSV rewrite disabled)",
        )
    return summary

