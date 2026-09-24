from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kstrbench.step import Step


@dataclass(kw_only=True)
class SelectCsvColumnsStep(Step):
    in_csv_file: str
    out_csv_file: str
    columns: list[str] = field(default_factory=list)
    csv_encoding: str = "utf-8-sig"

    def run(self) -> dict[str, Any]:
        in_csv = Path(self.in_csv_file).expanduser().resolve()
        out_csv = Path(self.out_csv_file).expanduser().resolve()
        if not in_csv.is_file():
            raise FileNotFoundError(f"in_csv_file not found: {in_csv}")
        if not self.columns:
            raise ValueError("columns is required")

        tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0

        with in_csv.open("r", encoding=self.csv_encoding, newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or [])
            missing = [name for name in self.columns if name not in fieldnames]
            if missing:
                raise ValueError(f"CSV missing columns {missing}: {in_csv}")
            with tmp_csv.open("w", encoding=self.csv_encoding, newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=self.columns, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    writer.writerow({name: row.get(name, "") for name in self.columns})
                    row_count += 1

        tmp_csv.replace(out_csv)
        return {
            "ok": True,
            "in_csv_file": str(in_csv),
            "out_csv_file": str(out_csv),
            "columns": list(self.columns),
            "row_count": row_count,
        }
