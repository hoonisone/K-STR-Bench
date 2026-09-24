from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kstrbench.step import Step


@dataclass(kw_only=True)
class ApplyTextUpdateCsvStep(Step):
    in_text_csv_file: str
    update_csv_file: str
    out_text_csv_file: str
    id_col: str = "text_id"
    split_col: str = "split"
    label_col: str = "label"
    csv_encoding: str = "utf-8-sig"

    def run(self) -> dict[str, Any]:
        in_csv = Path(self.in_text_csv_file).expanduser().resolve()
        update_csv = Path(self.update_csv_file).expanduser().resolve()
        out_csv = Path(self.out_text_csv_file).expanduser().resolve()
        if not in_csv.is_file():
            raise FileNotFoundError(f"in_text_csv_file not found: {in_csv}")
        if not update_csv.is_file():
            raise FileNotFoundError(f"update_csv_file not found: {update_csv}")
        if not self.id_col:
            raise ValueError("id_col is required")

        updates: dict[str, dict[str, str]] = {}
        with update_csv.open("r", encoding=self.csv_encoding, newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            if self.id_col not in fields:
                raise ValueError(f"update CSV missing '{self.id_col}': {update_csv}")
            for row in reader:
                tid = str(row.get(self.id_col, "") or "").strip()
                if tid:
                    updates[tid] = dict(row)

        row_count = 0
        matched = 0
        split_updated = 0
        label_updated = 0
        unmatched_update = 0

        tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)

        with in_csv.open("r", encoding=self.csv_encoding, newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or [])
            if self.id_col not in fieldnames:
                raise ValueError(f"text CSV missing '{self.id_col}': {in_csv}")
            out_fields = list(fieldnames)
            if self.split_col and self.split_col not in out_fields:
                out_fields.append(self.split_col)
            if self.label_col and self.label_col not in out_fields:
                out_fields.append(self.label_col)

            seen: set[str] = set()
            with tmp_csv.open("w", encoding=self.csv_encoding, newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=out_fields, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    row_count += 1
                    tid = str(row.get(self.id_col, "") or "").strip()
                    patch = updates.get(tid)
                    if patch is not None:
                        matched += 1
                        seen.add(tid)
                        if self.split_col and self.split_col in patch:
                            new_split = str(patch.get(self.split_col, "") or "")
                            if new_split != str(row.get(self.split_col, "") or ""):
                                split_updated += 1
                            row[self.split_col] = new_split
                        if self.label_col:
                            new_label = str(patch.get(self.label_col, "") or "")
                            if new_label:
                                if new_label != str(row.get(self.label_col, "") or ""):
                                    label_updated += 1
                                row[self.label_col] = new_label
                    writer.writerow({name: row.get(name, "") for name in out_fields})

        unmatched_update = len(set(updates) - seen)
        tmp_csv.replace(out_csv)
        return {
            "ok": True,
            "in_text_csv_file": str(in_csv),
            "update_csv_file": str(update_csv),
            "out_text_csv_file": str(out_csv),
            "row_count": row_count,
            "update_row_count": len(updates),
            "matched_count": matched,
            "split_updated_count": split_updated,
            "label_updated_count": label_updated,
            "unmatched_update_count": unmatched_update,
        }
