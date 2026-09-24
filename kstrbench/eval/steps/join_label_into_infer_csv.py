"""
Copy ground-truth labels into an inference CSV by matching text_id.

The label CSV (e.g. text.csv) is the source of truth. Rows of the inference
CSV whose text_id is found there receive the label in label_col.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from kstrbench.step import Step

DEFAULT_CSV_ENCODING = "utf-8-sig"

_LABEL_MAP_CACHE: Dict[tuple[str, str, str, str], Dict[str, str]] = {}


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def load_label_map(
    label_csv_path: Path,
    text_id_col: str,
    label_col: str,
    csv_encoding: str,
) -> Dict[str, str]:
    cache_key = (str(label_csv_path), text_id_col, label_col, csv_encoding)
    cached = _LABEL_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not label_csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv_path}")

    with label_csv_path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {label_csv_path}")
        missing = {text_id_col, label_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Label CSV missing columns {sorted(missing)}: {label_csv_path}. "
                f"Available: {reader.fieldnames}"
            )

        label_map: Dict[str, str] = {}
        for row in reader:
            text_id = str(row.get(text_id_col, "")).strip()
            if is_empty_value(text_id):
                continue
            label_map[text_id] = str(row.get(label_col, "") or "")

    _LABEL_MAP_CACHE[cache_key] = label_map
    return label_map


def execute(
    label_csv_path: Path,
    infer_csv_path: Path,
    output_csv_path: Path,
    text_id_col: str = "text_id",
    label_col: str = "label",
    csv_encoding: str = DEFAULT_CSV_ENCODING,
) -> Dict[str, Any]:
    if not infer_csv_path.exists():
        raise FileNotFoundError(f"Infer CSV not found: {infer_csv_path}")

    label_map = load_label_map(label_csv_path, text_id_col, label_col, csv_encoding)

    with infer_csv_path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {infer_csv_path}")
        if text_id_col not in reader.fieldnames:
            raise ValueError(
                f"Infer CSV missing column '{text_id_col}': {infer_csv_path}. "
                f"Available: {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        if label_col not in fieldnames:
            fieldnames.append(label_col)
        rows: List[Dict[str, Any]] = list(reader)

    matched_count = 0
    unmatched_count = 0
    for row in rows:
        text_id = str(row.get(text_id_col, "")).strip()
        label = label_map.get(text_id)
        if label is None:
            unmatched_count += 1
            row[label_col] = ""
            continue
        matched_count += 1
        row[label_col] = label

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding=csv_encoding, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"label_csv_path  : {label_csv_path}")
    print(f"infer_csv_path  : {infer_csv_path}")
    print(f"output_csv_path : {output_csv_path}")
    print(f"text_id_col     : {text_id_col}")
    print(f"label_col       : {label_col}")
    print(f"label_map_size  : {len(label_map)}")
    print(f"rows            : {len(rows)}")
    print(f"matched         : {matched_count}")
    print(f"unmatched       : {unmatched_count}")

    return {
        "label_csv_path": str(label_csv_path),
        "infer_csv_path": str(infer_csv_path),
        "output_csv_path": str(output_csv_path),
        "text_id_col": text_id_col,
        "label_col": label_col,
        "csv_encoding": csv_encoding,
        "label_map_size": len(label_map),
        "rows": len(rows),
        "matched": matched_count,
        "unmatched": unmatched_count,
    }


@dataclass(kw_only=True)
class JoinLabelIntoInferCsvStep(Step):
    label_csv_path: str
    infer_csv_path: str
    output_csv_path: str
    text_id_col: str = "text_id"
    label_col: str = "label"
    csv_encoding: str = DEFAULT_CSV_ENCODING

    def run(self) -> Dict[str, Any]:
        return execute(
            label_csv_path=Path(self.label_csv_path),
            infer_csv_path=Path(self.infer_csv_path),
            output_csv_path=Path(self.output_csv_path),
            text_id_col=self.text_id_col,
            label_col=self.label_col,
            csv_encoding=self.csv_encoding,
        )
