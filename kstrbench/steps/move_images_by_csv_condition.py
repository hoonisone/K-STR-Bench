"""
Move or copy images for CSV rows that match column-value filters.

By default, destination filenames use the basename of image_path_col
(without intermediate directories). Set unique_col_name to rename files
from another CSV column (e.g. text_id).
"""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from kstrbench.step import Step

DEFAULT_CSV_ENCODING = "utf-8-sig"


def normalize_filter_map(filters: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not filters:
        return {}
    return {str(key): value for key, value in filters.items()}


def non_conflict_destination(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    stem = dst_path.stem
    suffix = dst_path.suffix
    parent = dst_path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def is_expected_match(actual_value: Any, expected_value: Any) -> bool:
    if isinstance(expected_value, bool):
        if is_empty_value(actual_value):
            return False
        actual = str(actual_value).strip().lower()
        if actual in {"true", "1", "yes", "t"}:
            return expected_value is True
        if actual in {"false", "0", "no", "f"}:
            return expected_value is False
        return False

    expected = str(expected_value).strip()
    if expected == "":
        return is_empty_value(actual_value)
    if is_empty_value(actual_value):
        return False
    return str(actual_value).strip() == expected


def is_row_match(row: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for column_key, expected_value in filters.items():
        if not is_expected_match(row.get(column_key, ""), expected_value):
            return False
    return True


def is_row_excluded(row: Dict[str, Any], exclude_filters: Dict[str, Any]) -> bool:
    if not exclude_filters:
        return False
    for column_key, expected_value in exclude_filters.items():
        if not is_expected_match(row.get(column_key, ""), expected_value):
            return False
    return True


def build_source_path(
    data_dir: Path,
    row: Dict[str, Any],
    image_path_col: str,
    sub_dir_col: str | None,
) -> Path | None:
    image_name = str(row.get(image_path_col, "")).replace("\\", "/").strip()
    if is_empty_value(image_name):
        return None

    if sub_dir_col is None:
        return data_dir / Path(image_name)

    sub_dir = str(row.get(sub_dir_col, "")).replace("\\", "/").strip()
    if is_empty_value(sub_dir):
        return None
    return data_dir / sub_dir / Path(image_name).name


def image_path_filename(
    row: Dict[str, Any],
    image_path_col: str,
    source_path: Path,
) -> str:
    image_rel_path = str(row.get(image_path_col, "")).replace("\\", "/").strip()
    if not is_empty_value(image_rel_path):
        filename = Path(image_rel_path).name
        if filename:
            return filename
    return source_path.name


def build_destination_path(
    new_data_dir: Path,
    source_path: Path,
    row: Dict[str, Any],
    image_path_col: str,
    unique_col_name: str | None,
) -> Path | None:
    if unique_col_name is None:
        destination_name = image_path_filename(row, image_path_col, source_path)
        return non_conflict_destination(new_data_dir / destination_name)

    unique_value = str(row.get(unique_col_name, "")).strip().replace("\\", "/")
    if is_empty_value(unique_value):
        return None

    unique_path = Path(unique_value)
    if unique_path.suffix:
        destination_name = unique_path.name
    else:
        destination_name = f"{unique_path.name}{source_path.suffix.lower()}"
    return non_conflict_destination(new_data_dir / destination_name)


def execute(
    csv_path: Path,
    data_dir: Path,
    new_data_dir: Path,
    column_value_filters: Mapping[str, Any],
    image_path_col_name: str = "image_path",
    exclude_column_value_filters: Mapping[str, Any] | None = None,
    operation: str = "copy",
    unique_col_name: str | None = None,
    sub_dir_col: str | None = None,
    csv_encoding: str = DEFAULT_CSV_ENCODING,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    image_path_col = str(image_path_col_name)
    unique_col = str(unique_col_name) if unique_col_name is not None else None
    sub_dir_col_name = str(sub_dir_col) if sub_dir_col is not None else None
    include_filters = normalize_filter_map(column_value_filters)
    exclude_filters = normalize_filter_map(exclude_column_value_filters)

    operation_name = str(operation).lower().strip()
    if operation_name not in {"move", "copy"}:
        raise ValueError("operation must be one of: move, copy")

    required_columns = {
        image_path_col,
        *include_filters.keys(),
        *exclude_filters.keys(),
    }
    if unique_col:
        required_columns.add(unique_col)
    if sub_dir_col_name:
        required_columns.add(sub_dir_col_name)

    matched_count = 0
    excluded_count = 0
    processed_count = 0
    skipped_count = 0
    skipped_empty_unique_count = 0
    skipped_empty_sub_dir_count = 0

    with csv_path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Input CSV missing columns {sorted(missing_columns)}: {csv_path}. "
                f"Available: {reader.fieldnames}"
            )

        new_data_dir.mkdir(parents=True, exist_ok=True)

        for row in reader:
            if not is_row_match(row, include_filters):
                continue
            matched_count += 1
            if is_row_excluded(row, exclude_filters):
                excluded_count += 1
                continue

            source_path = build_source_path(
                data_dir=data_dir,
                row=row,
                image_path_col=image_path_col,
                sub_dir_col=sub_dir_col_name,
            )
            if source_path is None:
                if sub_dir_col_name is not None and is_empty_value(
                    row.get(sub_dir_col_name, "")
                ):
                    skipped_empty_sub_dir_count += 1
                else:
                    skipped_count += 1
                continue
            if not source_path.exists():
                skipped_count += 1
                continue

            destination_path = build_destination_path(
                new_data_dir=new_data_dir,
                source_path=source_path,
                row=row,
                image_path_col=image_path_col,
                unique_col_name=unique_col,
            )
            if destination_path is None:
                skipped_empty_unique_count += 1
                continue

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if operation_name == "copy":
                shutil.copy2(source_path, destination_path)
            else:
                shutil.move(source_path, destination_path)
            processed_count += 1

    print(f"csv_path                     : {csv_path}")
    print(f"image_path_col_name          : {image_path_col}")
    print(f"data_dir                     : {data_dir}")
    print(f"new_data_dir                 : {new_data_dir}")
    print(f"operation                    : {operation_name}")
    print(f"unique_col_name              : {unique_col}")
    print(f"sub_dir_col                  : {sub_dir_col_name}")
    print(f"column_value_filters         : {include_filters}")
    print(f"exclude_column_value_filters : {exclude_filters}")
    print(f"matched                      : {matched_count}")
    print(f"excluded                     : {excluded_count}")
    print(f"processed                    : {processed_count}")
    print(f"skipped                      : {skipped_count}")
    print(f"skipped_empty_unique         : {skipped_empty_unique_count}")
    print(f"skipped_empty_sub_dir        : {skipped_empty_sub_dir_count}")

    return {
        "csv_path": str(csv_path),
        "image_path_col_name": image_path_col,
        "data_dir": str(data_dir),
        "new_data_dir": str(new_data_dir),
        "operation": operation_name,
        "unique_col_name": unique_col,
        "sub_dir_col": sub_dir_col_name,
        "column_value_filters": include_filters,
        "exclude_column_value_filters": exclude_filters,
        "matched": matched_count,
        "excluded": excluded_count,
        "processed": processed_count,
        "skipped": skipped_count,
        "skipped_empty_unique": skipped_empty_unique_count,
        "skipped_empty_sub_dir": skipped_empty_sub_dir_count,
    }


@dataclass(kw_only=True)
class MoveImagesByCsvConditionStep(Step):
    csv_path: str
    data_dir: str
    new_data_dir: str
    column_value_filters: Dict[str, Any] = field(default_factory=dict)
    image_path_col: str = "image_path"
    exclude_column_value_filters: Dict[str, Any] = field(default_factory=dict)
    operation: str = "copy"
    unique_col_name: str | None = None
    sub_dir_col: str | None = None
    csv_encoding: str = DEFAULT_CSV_ENCODING

    def run(self) -> Dict[str, Any]:
        return execute(
            csv_path=Path(self.csv_path),
            data_dir=Path(self.data_dir),
            new_data_dir=Path(self.new_data_dir),
            column_value_filters=self.column_value_filters,
            image_path_col_name=self.image_path_col,
            exclude_column_value_filters=self.exclude_column_value_filters,
            operation=self.operation,
            unique_col_name=self.unique_col_name,
            sub_dir_col=self.sub_dir_col,
            csv_encoding=self.csv_encoding,
        )
