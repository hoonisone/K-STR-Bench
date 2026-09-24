"""
Fill a CSV column for rows whose text_id matches an image filename.

Each image under image_dir is indexed by stem (filename without extension).
CSV rows whose text_id matches receive output_value in output_col.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Set

from kstrbench.step import Step

DEFAULT_CSV_ENCODING = "utf-8-sig"
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_image_extensions(extensions: Set[str] | None) -> Set[str]:
    if extensions is None:
        return set(DEFAULT_IMAGE_EXTENSIONS)
    normalized = {str(ext).lower().strip() for ext in extensions}
    normalized = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in normalized
        if ext
    }
    if not normalized:
        raise ValueError("image_extensions must contain at least one extension.")
    return normalized


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return ""
    return str(value)


def load_text_ids_from_image_dir(
    image_dir: Path,
    image_extensions: Set[str],
) -> Set[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    text_ids: Set[str] = set()
    image_count = 0
    for image_path in image_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_extensions:
            continue
        stem = image_path.stem.strip()
        if not stem:
            continue
        image_count += 1
        text_ids.add(stem)

    if image_count == 0:
        raise ValueError(f"No image files found under: {image_dir}")
    return text_ids


def execute(
    csv_path: Path,
    image_dir: Path,
    output_col: str,
    output_value: Any,
    output_csv_path: Path | None = None,
    text_id_col: str = "text_id",
    csv_encoding: str = DEFAULT_CSV_ENCODING,
    overwrite_existing: bool = True,
    image_extensions: Set[str] | None = None,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dest_csv_path = output_csv_path or csv_path
    text_id_col_name = str(text_id_col)
    output_col_name = str(output_col)
    cell_value = format_cell(output_value)
    normalized_extensions = normalize_image_extensions(image_extensions)
    image_text_ids = load_text_ids_from_image_dir(image_dir, normalized_extensions)

    with csv_path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if text_id_col_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{text_id_col_name}' not found in {csv_path}. "
                f"Available: {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        if output_col_name not in fieldnames:
            fieldnames.append(output_col_name)
        rows: list[Dict[str, Any]] = list(reader)

    matched_count = 0
    unmatched_count = 0
    skipped_existing_count = 0

    for row in rows:
        text_id = str(row.get(text_id_col_name, "")).strip()
        if is_empty_value(text_id) or text_id not in image_text_ids:
            unmatched_count += 1
            continue
        if (
            not overwrite_existing
            and output_col_name in row
            and not is_empty_value(row.get(output_col_name, ""))
        ):
            skipped_existing_count += 1
            continue
        row[output_col_name] = cell_value
        matched_count += 1

    dest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv_path.open("w", encoding=csv_encoding, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"csv_path               : {csv_path}")
    print(f"output_csv_path        : {dest_csv_path}")
    print(f"image_dir              : {image_dir}")
    print(f"text_id_col            : {text_id_col_name}")
    print(f"output_col             : {output_col_name}")
    print(f"output_value           : {cell_value}")
    print(f"image_file_count       : {len(image_text_ids)}")
    print(f"csv_row_count          : {len(rows)}")
    print(f"matched_count          : {matched_count}")
    print(f"unmatched_count        : {unmatched_count}")
    print(f"skipped_existing_count : {skipped_existing_count}")

    return {
        "csv_path": str(csv_path),
        "output_csv_path": str(dest_csv_path),
        "image_dir": str(image_dir),
        "text_id_col": text_id_col_name,
        "output_col": output_col_name,
        "output_value": cell_value,
        "csv_encoding": csv_encoding,
        "overwrite_existing": overwrite_existing,
        "image_file_count": len(image_text_ids),
        "csv_row_count": len(rows),
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "skipped_existing_count": skipped_existing_count,
    }


@dataclass(kw_only=True)
class FillCsvColumnFromImageFilenamesStep(Step):
    csv_path: str
    image_dir: str
    output_col: str
    output_value: Any
    output_csv_path: str | None = None
    text_id_col: str = "text_id"
    csv_encoding: str = DEFAULT_CSV_ENCODING
    overwrite_existing: bool = True
    image_extensions: Set[str] | None = None

    def run(self) -> Dict[str, Any]:
        return execute(
            csv_path=Path(self.csv_path),
            image_dir=Path(self.image_dir),
            output_col=self.output_col,
            output_value=self.output_value,
            output_csv_path=Path(self.output_csv_path) if self.output_csv_path else None,
            text_id_col=self.text_id_col,
            csv_encoding=self.csv_encoding,
            overwrite_existing=self.overwrite_existing,
            image_extensions=self.image_extensions,
        )
