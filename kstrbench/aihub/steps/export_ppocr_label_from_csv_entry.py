from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class ExportPPOCRLabelFromCsvStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file")
    image_path_col = str(params.get("image_path_col", "image_path")).strip()
    label_col = str(params.get("label_col", "text")).strip()
    out_txt_file_raw = params.get("out_txt_file")
    skip_empty_image_path = coerce_bool(params.get("skip_empty_image_path", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_txt_file_raw:
        raise ValueError("out_txt_file is required")
    if not image_path_col:
        raise ValueError("image_path_col is required")
    if not label_col:
        raise ValueError("label_col is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_txt_file = Path(str(out_txt_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if image_path_col not in fieldnames:
            raise ValueError(f"CSV missing image_path_col '{image_path_col}': {in_csv_file}")
        if label_col not in fieldnames:
            raise ValueError(f"CSV missing label_col '{label_col}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    out_txt_file.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    skipped_count = 0
    with out_txt_file.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            img = str(row.get(image_path_col, "") or "").strip()
            txt = str(row.get(label_col, "") or "")
            if not img and skip_empty_image_path:
                skipped_count += 1
                continue
            f.write(f"{img}\t{txt}\n")
            written_count += 1

    logger = StepLogger("export_ppocr_label_from_csv_entry")
    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_txt_file": str(out_txt_file),
        "image_path_col": image_path_col,
        "label_col": label_col,
        "row_count": len(rows),
        "written_count": written_count,
        "skipped_count": skipped_count,
        "skip_empty_image_path": skip_empty_image_path,
        "log_level": log_level,
    }
    logger.log(summary, title="export ppocr label from csv summary")
    emit(log_level, "INFO", f"[ppocr-export] rows={len(rows)}, written={written_count}, skipped={skipped_count}")
    emit(log_level, "INFO", f"[ppocr-export] out_txt={out_txt_file}")
    return summary

