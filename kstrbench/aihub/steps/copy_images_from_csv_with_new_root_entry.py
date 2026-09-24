from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class CopyImagesFromCsvWithNewRootStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _normalize_rel_path(raw: str) -> str:
    return raw.strip().replace("\\", "/").lstrip("/")


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

    in_csv_file_raw = params.get("in_csv_file")
    image_path_col = str(params.get("image_path_col", "image_path")).strip()
    source_root_dir_raw = params.get("source_root_dir")
    root_dir_raw = params.get("target_dir", params.get("root_dir", params.get("target_root_dir")))
    filters = _parse_filters(params.get("filters"))
    allow_missing_files = coerce_bool(params.get("allow_missing_files", False), default=False)
    overwrite = coerce_bool(params.get("overwrite", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not image_path_col:
        raise ValueError("image_path_col is required")
    if not source_root_dir_raw:
        raise ValueError("source_root_dir is required")
    if not root_dir_raw:
        raise ValueError("root_dir (or target_root_dir) is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    source_root_dir = Path(str(source_root_dir_raw)).expanduser().resolve()
    root_dir = Path(str(root_dir_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")
    if not source_root_dir.is_dir():
        raise FileNotFoundError(f"source_root_dir not found: {source_root_dir}")
    root_dir.mkdir(parents=True, exist_ok=True)

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if image_path_col not in fieldnames:
            raise ValueError(f"CSV missing image_path_col '{image_path_col}': {in_csv_file}")
        for flt in filters:
            if str(flt["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{flt['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    filtered_rows = rows
    if filters:
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

    row_iter: Any = (
        tqdm(filtered_rows, desc="copy images by csv", unit="row", dynamic_ncols=True)
        if show_progress
        else filtered_rows
    )

    copied_count = 0
    skipped_existing_count = 0
    missing_count = 0
    error_count = 0
    for row in row_iter:
        raw_path = str(row.get(image_path_col, "") or "").strip()
        if not raw_path:
            continue
        rel_path = _normalize_rel_path(raw_path)
        src_abs = (source_root_dir / rel_path).resolve()
        dst_abs = (root_dir / Path(rel_path).name).resolve()

        if not src_abs.is_file():
            if allow_missing_files:
                missing_count += 1
                continue
            raise FileNotFoundError(f"source image not found: {src_abs}")

        if dst_abs.exists() and not overwrite:
            skipped_existing_count += 1
            continue
        try:
            if dst_abs.exists() and overwrite:
                dst_abs.unlink()
            shutil.copy2(src_abs, dst_abs)
            copied_count += 1
        except Exception:
            error_count += 1
            if not allow_missing_files:
                raise

    logger = StepLogger("copy_images_from_csv_with_new_root_entry")
    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "image_path_col": image_path_col,
        "filters": filters,
        "source_root_dir": str(source_root_dir),
        "root_dir": str(root_dir),
        "input_row_count": len(rows),
        "row_count": len(filtered_rows),
        "copied_count": copied_count,
        "skipped_existing_count": skipped_existing_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "allow_missing_files": allow_missing_files,
        "overwrite": overwrite,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(summary, title="copy images from csv with new root summary")
    emit(
        log_level,
        "INFO",
        (
            f"[copy-images] rows={len(filtered_rows)}, copied={copied_count}, skipped={skipped_existing_count}, "
            f"missing={missing_count}, errors={error_count}"
        ),
    )
    emit(log_level, "INFO", f"[copy-images] root_dir={root_dir}")
    return summary

