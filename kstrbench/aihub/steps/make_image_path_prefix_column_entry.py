from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_path_text(text: str) -> str:
    return str(text).strip().replace("\\", "/")


def _make_prefix(path_text: str, token_count: int) -> str:
    norm = _normalize_path_text(path_text).strip("/")
    if not norm:
        return ""
    parts = [p for p in norm.split("/") if p]
    if token_count <= 0:
        return ""
    return "/".join(parts[:token_count])


class MakeImagePathPrefixColumnStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file")
    out_csv_file_raw = params.get("out_csv_file", in_csv_file_raw)
    path_column = str(params.get("path_column", "image_file"))
    out_column = str(params.get("out_column", "image_path_prefix_4"))
    prefix_token_count = int(params.get("prefix_token_count", 4))
    overwrite_existing = coerce_bool(params.get("overwrite_existing", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file is required")
    if not path_column:
        raise ValueError("path_column is required")
    if not out_column:
        raise ValueError("out_column is required")
    if prefix_token_count < 0:
        raise ValueError("prefix_token_count must be >= 0")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("make_image_path_prefix_column_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "path_column": path_column,
            "out_column": out_column,
            "prefix_token_count": prefix_token_count,
            "overwrite_existing": overwrite_existing,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="make image path prefix discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if path_column not in fieldnames:
            raise ValueError(f"CSV must include '{path_column}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    out_fieldnames = list(fieldnames)
    if out_column not in out_fieldnames:
        out_fieldnames.append(out_column)

    updated_count = 0
    skipped_existing_count = 0

    row_iter: Any = (
        tqdm(rows, desc="make image path prefix", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )
    out_rows: list[dict[str, str]] = []
    for idx, row in enumerate(row_iter):
        out = dict(row)
        existing = str(out.get(out_column, "") or "").strip()
        if existing and (not overwrite_existing):
            skipped_existing_count += 1
            out_rows.append(out)
            continue

        path_value = str(out.get(path_column, "") or "")
        try:
            out[out_column] = _make_prefix(path_value, prefix_token_count)
            updated_count += 1
            out_rows.append(out)
        except Exception as exc:
            error_info = {
                "row_index": idx,
                "path_column": path_column,
                "path_value": path_value,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            logger.log(error_info, title="make image path prefix row error")
            emit(
                log_level,
                "ERROR",
                (
                    f"[path-prefix][ERROR] row={idx}, path_column={path_column}, "
                    f"path='{path_value}', error={type(exc).__name__}: {exc}"
                ),
            )
            raise

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    result = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "path_column": path_column,
        "out_column": out_column,
        "prefix_token_count": prefix_token_count,
        "overwrite_existing": overwrite_existing,
        "show_progress": show_progress,
        "log_level": log_level,
        "row_count": len(rows),
        "updated_count": updated_count,
        "skipped_existing_count": skipped_existing_count,
    }
    logger.log(result, title="make image path prefix summary")
    emit(
        log_level,
        "INFO",
        (
            f"[path-prefix] rows={len(rows)}, updated={updated_count}, "
            f"skipped_existing={skipped_existing_count}"
        ),
    )
    emit(log_level, "INFO", f"[path-prefix] out_csv={out_csv_file}")
    return result

