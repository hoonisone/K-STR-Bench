from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..columns import SCENE_ID_PREFIX_AIHUB
from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_path_text(text: str) -> str:
    return text.strip().replace("\\", "/")


def _resolve_abs_path(dataset_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p.expanduser().resolve() if p.is_absolute() else (dataset_dir / p).resolve()


def _extract_image_id_from_path(image_path: str) -> int:
    stem = Path(image_path).stem
    if stem.isdigit():
        return int(stem)
    prefix = f"{SCENE_ID_PREFIX_AIHUB}_"
    if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
        return int(stem[len(prefix) :])
    raise ValueError(f"image filename stem is not a numeric or {prefix}* id: {image_path}")


def _parse_parallelism(params: dict[str, Any]) -> tuple[int, int]:
    p_raw = params.get("num_process", params.get("num_proc", 1))
    t_raw = params.get("num_thread", params.get("num_threads", params.get("num_thr", 1)))
    num_process = max(1, int(p_raw))
    num_thread = max(1, int(t_raw))
    return num_process, num_thread


def _coerce_int_for_sort(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunk_tasks(tasks: list[tuple[int, dict[str, str]]], n: int) -> list[list[tuple[int, dict[str, str]]]]:
    if n <= 1 or len(tasks) <= 1:
        return [tasks]
    chunks: list[list[tuple[int, dict[str, str]]]] = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        chunks[i % n].append(task)
    return [c for c in chunks if c]


def _parse_one_pair(
    task: tuple[int, dict[str, str]],
    dataset_dir_str: str,
    image_col: str,
    label_col: str,
    annotation_image_id_col: str,
    annotation_idx_col: str,
    skip_missing_files: bool,
) -> dict[str, Any]:
    idx, row = task
    dataset_dir = Path(dataset_dir_str)
    image_path = _normalize_path_text(str(row.get(image_col, "") or ""))
    label_path = _normalize_path_text(str(row.get(label_col, "") or ""))
    if not image_path or not label_path:
        return {"status": "skipped_empty", "idx": idx}

    image_id = _extract_image_id_from_path(image_path)
    label_abs = _resolve_abs_path(dataset_dir, label_path)
    if not label_abs.is_file():
        if skip_missing_files:
            return {"status": "missing_label", "idx": idx}
        raise FileNotFoundError(f"label file not found: {label_abs}")

    try:
        with label_abs.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"status": "parse_error", "idx": idx}

    image_info: dict[str, Any] = {}
    images_raw = payload.get("images", [])
    if isinstance(images_raw, list) and images_raw:
        first = images_raw[0]
        if isinstance(first, dict):
            image_info = first

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    image_row: dict[str, Any] = {
        "image_id": image_id,
        image_col: image_path,
        label_col: label_path,
        "width": image_info.get("width"),
        "height": image_info.get("height"),
        "data_captured_date": image_info.get("date_captured"),
    }
    for k, v in metadata.items():
        image_row[str(k)] = v

    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    annotation_rows: list[dict[str, Any]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        ann_idx = ann.get("id")
        if ann_idx is None:
            continue
        ann_idx_int = _coerce_int_for_sort(ann_idx, default=0)
        annotation_rows.append(
            {
                annotation_image_id_col: image_id,
                annotation_idx_col: ann_idx,
                "_ann_sort_idx": ann_idx_int,
                "text": ann.get("text"),
                "bbox": json.dumps(ann.get("bbox", []), ensure_ascii=False),
            }
        )

    return {
        "status": "ok",
        "idx": idx,
        "image_row": image_row,
        "annotation_rows": annotation_rows,
        "metadata_keys": list(metadata.keys()),
    }


def _run_batch(
    tasks: list[tuple[int, dict[str, str]]],
    dataset_dir_str: str,
    image_col: str,
    label_col: str,
    annotation_image_id_col: str,
    annotation_idx_col: str,
    skip_missing_files: bool,
    num_thread: int,
    show_progress: bool = False,
    progress_desc: str = "export pair metadata p0:t0",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if show_progress:
        # show progress only for first process / first thread lane
        num_thread = 1
    if num_thread <= 1:
        task_iter: Any = (
            tqdm(tasks, desc=progress_desc, unit="pair", dynamic_ncols=True)
            if show_progress
            else tasks
        )
        for task in task_iter:
            results.append(
                _parse_one_pair(
                    task=task,
                    dataset_dir_str=dataset_dir_str,
                    image_col=image_col,
                    label_col=label_col,
                    annotation_image_id_col=annotation_image_id_col,
                    annotation_idx_col=annotation_idx_col,
                    skip_missing_files=skip_missing_files,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=num_thread) as ex:
            futs = [
                ex.submit(
                    _parse_one_pair,
                    task,
                    dataset_dir_str,
                    image_col,
                    label_col,
                    annotation_image_id_col,
                    annotation_idx_col,
                    skip_missing_files,
                )
                for task in tasks
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    image_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    metadata_key_set: set[str] = set()
    missing_label_count = 0
    parse_error_count = 0
    skipped_count = 0

    for result in results:
        status = result.get("status")
        if status == "ok":
            image_rows.append(result["image_row"])
            annotation_rows.extend(result["annotation_rows"])
            metadata_key_set.update(str(k) for k in result.get("metadata_keys", []))
        elif status == "missing_label":
            missing_label_count += 1
        elif status == "parse_error":
            parse_error_count += 1
        elif status == "skipped_empty":
            skipped_count += 1

    return {
        "image_rows": image_rows,
        "annotation_rows": annotation_rows,
        "metadata_keys": sorted(metadata_key_set),
        "missing_label_count": missing_label_count,
        "parse_error_count": parse_error_count,
        "skipped_count": skipped_count,
    }


class ExportImageLabelPairMetadataCsvStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    match_pair_csv_file_raw = params.get("match_pair_csv_file")
    out_image_csv_file_raw = params.get("out_image_csv_file")
    out_annotation_csv_file_raw = params.get("out_annotation_csv_file")
    image_col = str(params.get("image_col", "image_file"))
    label_col = str(params.get("label_col", "label_file"))
    annotation_image_id_col = str(params.get("annotation_image_id_col", "image_id"))
    annotation_idx_col = str(params.get("annotation_idx_col", "annotation_idx"))
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    skip_missing_files = coerce_bool(params.get("skip_missing_files", True), default=True)
    sort_metadata_keys = coerce_bool(params.get("sort_metadata_keys", True), default=True)
    num_process, num_thread = _parse_parallelism(params)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not dataset_dir_raw:
        raise ValueError("dataset_dir (or datadir) is required")
    if not match_pair_csv_file_raw:
        raise ValueError("match_pair_csv_file is required")
    if not out_image_csv_file_raw:
        raise ValueError("out_image_csv_file is required")
    if not out_annotation_csv_file_raw:
        raise ValueError("out_annotation_csv_file is required")

    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    match_pair_csv_file = Path(str(match_pair_csv_file_raw)).expanduser().resolve()
    out_image_csv_file = Path(str(out_image_csv_file_raw)).expanduser().resolve()
    out_annotation_csv_file = Path(str(out_annotation_csv_file_raw)).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    if not match_pair_csv_file.is_file():
        raise FileNotFoundError(f"match_pair_csv_file not found: {match_pair_csv_file}")

    logger = StepLogger("export_image_label_pair_metadata_csv_entry")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "match_pair_csv_file": str(match_pair_csv_file),
            "out_image_csv_file": str(out_image_csv_file),
            "out_annotation_csv_file": str(out_annotation_csv_file),
            "image_col": image_col,
            "label_col": label_col,
            "annotation_image_id_col": annotation_image_id_col,
            "annotation_idx_col": annotation_idx_col,
            "show_progress": show_progress,
            "skip_missing_files": skip_missing_files,
            "sort_metadata_keys": sort_metadata_keys,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
        },
        title="pair metadata export discovery",
    )

    with match_pair_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if image_col not in fieldnames or label_col not in fieldnames:
            raise ValueError(
                f"CSV must include '{image_col}' and '{label_col}': {match_pair_csv_file}"
            )
        pair_rows = [dict(row) for row in reader]

    tasks: list[tuple[int, dict[str, str]]] = [(i, row) for i, row in enumerate(pair_rows)]
    task_chunks = _chunk_tasks(tasks, num_process)

    image_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    metadata_key_set: set[str] = set()
    missing_label_count = 0
    parse_error_count = 0
    skipped_count = 0

    if num_process <= 1:
        result = _run_batch(
            tasks=task_chunks[0],
            dataset_dir_str=str(dataset_dir),
            image_col=image_col,
            label_col=label_col,
            annotation_image_id_col=annotation_image_id_col,
            annotation_idx_col=annotation_idx_col,
            skip_missing_files=skip_missing_files,
            num_thread=num_thread,
            show_progress=show_progress,
            progress_desc="export pair metadata p0:t0",
        )
        image_rows.extend(result["image_rows"])
        annotation_rows.extend(result["annotation_rows"])
        metadata_key_set.update(result["metadata_keys"])
        missing_label_count += int(result["missing_label_count"])
        parse_error_count += int(result["parse_error_count"])
        skipped_count += int(result["skipped_count"])
    else:
        with ProcessPoolExecutor(max_workers=num_process) as ex:
            futs = [
                ex.submit(
                    _run_batch,
                    chunk,
                    str(dataset_dir),
                    image_col,
                    label_col,
                    annotation_image_id_col,
                    annotation_idx_col,
                    skip_missing_files,
                    num_thread,
                    show_progress and i == 0,
                    "export pair metadata p0:t0",
                )
                for i, chunk in enumerate(task_chunks)
            ]
            for fut in as_completed(futs):
                result = fut.result()
                image_rows.extend(result["image_rows"])
                annotation_rows.extend(result["annotation_rows"])
                metadata_key_set.update(result["metadata_keys"])
                missing_label_count += int(result["missing_label_count"])
                parse_error_count += int(result["parse_error_count"])
                skipped_count += int(result["skipped_count"])

    image_rows.sort(key=lambda r: _coerce_int_for_sort(r.get("image_id"), default=0))
    annotation_rows.sort(
        key=lambda r: (
            _coerce_int_for_sort(r.get(annotation_image_id_col), default=0),
            _coerce_int_for_sort(r.get("_ann_sort_idx"), default=0),
        )
    )
    for row in annotation_rows:
        row.pop("_ann_sort_idx", None)

    metadata_cols = sorted(metadata_key_set) if sort_metadata_keys else list(metadata_key_set)
    image_header = [
        "image_id",
        image_col,
        label_col,
        "width",
        "height",
        "data_captured_date",
    ] + metadata_cols
    annotation_header = [
        annotation_image_id_col,
        annotation_idx_col,
        "text",
        "bbox",
    ]

    out_image_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_image_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=image_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)

    out_annotation_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_annotation_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotation_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annotation_rows)

    summary = {
        "ok": True,
        "dataset_dir": str(dataset_dir),
        "match_pair_csv_file": str(match_pair_csv_file),
        "out_image_csv_file": str(out_image_csv_file),
        "out_annotation_csv_file": str(out_annotation_csv_file),
        "image_col": image_col,
        "label_col": label_col,
        "annotation_image_id_col": annotation_image_id_col,
        "annotation_idx_col": annotation_idx_col,
        "pair_count": len(pair_rows),
        "image_row_count": len(image_rows),
        "annotation_row_count": len(annotation_rows),
        "metadata_column_count": len(metadata_cols),
        "missing_label_count": missing_label_count,
        "parse_error_count": parse_error_count,
        "skipped_count": skipped_count,
        "show_progress": show_progress,
        "skip_missing_files": skip_missing_files,
        "sort_metadata_keys": sort_metadata_keys,
        "num_process": num_process,
        "num_thread": num_thread,
        "log_level": log_level,
    }
    logger.log(summary, title="pair metadata export summary")
    emit(log_level, "INFO", f"[pair-metadata] pair_csv={match_pair_csv_file}")
    emit(log_level, "INFO", f"[pair-metadata] image_csv={out_image_csv_file}")
    emit(log_level, "INFO", f"[pair-metadata] annotations_csv={out_annotation_csv_file}")
    emit(
        log_level,
        "INFO",
        (
            f"[pair-metadata] pairs={len(pair_rows)}, images={len(image_rows)}, "
            f"annotations={len(annotation_rows)}, missing_label={missing_label_count}, parse_error={parse_error_count}"
        ),
    )
    return summary
