from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class AddTextLanguageCategoryColumnStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _is_korean(ch: str) -> bool:
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
    )


def _is_english(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_special(ch: str) -> bool:
    if _is_korean(ch) or _is_english(ch) or ch.isdigit() or ch.isspace():
        return False
    return True


def _classify_text(text: str, ignore_space: bool = True) -> str:
    categories: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch.isspace():
            if ignore_space:
                continue
            cat = "SPACE"
            if cat not in seen:
                seen.add(cat)
                categories.append(cat)
            continue
        if _is_korean(ch):
            cat = "KOR"
        elif _is_english(ch):
            cat = "ENG"
        elif ch.isdigit():
            cat = "DIGIT"
        elif _is_special(ch):
            cat = "SPECIAL"
        else:
            cat = "OTHERS"
        if cat not in seen:
            seen.add(cat)
            categories.append(cat)
    if not categories:
        return "OTHERS"
    return "+".join(categories)


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file", params.get("annotation_csv_file"))
    out_csv_file_raw = params.get("out_csv_file", params.get("out_annotation_csv_file", in_csv_file_raw))
    label_column = str(params.get("label_column", "text")).strip()
    out_column = str(params.get("out_column", "language_category")).strip()
    ignore_space = coerce_bool(params.get("ignore_space", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file (or annotation_csv_file) is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file (or out_annotation_csv_file) is required")
    if not label_column:
        raise ValueError("label_column is required")
    if not out_column:
        raise ValueError("out_column is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("add_text_language_category_column_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "label_column": label_column,
            "out_column": out_column,
            "ignore_space": ignore_space,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="add text language category column discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if label_column not in fieldnames:
            raise ValueError(f"CSV missing label_column '{label_column}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    row_iter: Any = (
        tqdm(rows, desc="add language category", unit="row", dynamic_ncols=True)
        if show_progress
        else rows
    )

    category_counts: dict[str, int] = {}
    for row in row_iter:
        text = str(row.get(label_column, "") or "")
        cat = _classify_text(text, ignore_space=ignore_space)
        row[out_column] = cat
        category_counts[cat] = category_counts.get(cat, 0) + 1

    out_fields = list(fieldnames)
    if out_column not in out_fields:
        out_fields.append(out_column)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "label_column": label_column,
        "out_column": out_column,
        "ignore_space": ignore_space,
        "row_count": len(rows),
        "category_counts": category_counts,
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(summary, title="add text language category column summary")
    emit(log_level, "INFO", f"[lang-cat] rows={len(rows)}, counts={category_counts}")
    emit(log_level, "INFO", f"[lang-cat] out_csv={out_csv_file}")
    return summary

