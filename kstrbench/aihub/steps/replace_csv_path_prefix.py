from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kstrbench.step import Step


def _replace_prefix(path_text: str, old_prefix: str, new_prefix: str) -> str:
    norm = path_text.strip().replace("\\", "/")
    old = old_prefix.strip().replace("\\", "/").rstrip("/")
    new = new_prefix.strip().replace("\\", "/").rstrip("/")
    if not norm or not old:
        return norm
    if norm == old:
        return new
    token = old + "/"
    if norm.startswith(token):
        return new + "/" + norm[len(token) :]
    return norm


@dataclass(kw_only=True)
class ReplaceCsvPathPrefixStep(Step):
    in_csv_file: str
    out_csv_file: str
    column: str
    old_prefix: str
    new_prefix: str
    csv_encoding: str = "utf-8-sig"

    def run(self) -> dict[str, Any]:
        in_csv = Path(self.in_csv_file).expanduser().resolve()
        out_csv = Path(self.out_csv_file).expanduser().resolve()
        if not in_csv.is_file():
            raise FileNotFoundError(f"in_csv_file not found: {in_csv}")
        if not self.column:
            raise ValueError("column is required")

        changed = 0
        total = 0
        tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)

        with in_csv.open("r", encoding=self.csv_encoding, newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or [])
            if self.column not in fieldnames:
                raise ValueError(f"CSV missing column '{self.column}': {in_csv}")
            with tmp_csv.open("w", encoding=self.csv_encoding, newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    total += 1
                    before = str(row.get(self.column, "") or "")
                    after = _replace_prefix(before, self.old_prefix, self.new_prefix)
                    if after != before.replace("\\", "/"):
                        changed += 1
                    row[self.column] = after
                    writer.writerow(row)

        tmp_csv.replace(out_csv)
        return {
            "ok": True,
            "in_csv_file": str(in_csv),
            "out_csv_file": str(out_csv),
            "column": self.column,
            "old_prefix": self.old_prefix,
            "new_prefix": self.new_prefix,
            "row_count": total,
            "changed_count": changed,
        }
