"""
Collect scored infer CSVs into one summary table.

One output row is a (regime, model, dataset, challenge) cell holding sample
count, word accuracy and jaso-level 1-NED. An empty challenge means the whole
dataset. A dataset may span several scored CSVs. Challenge tags for that
dataset are the union of the text CSVs for those sources.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from kstrbench.step import Step

DEFAULT_CSV_ENCODING = "utf-8-sig"
TRUE_VALUES = {"true", "1", "yes", "t"}


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def load_tagged_ids(
    tag_csv_paths: Sequence[Path],
    challenge_cols: Sequence[str],
    text_id_col: str,
    csv_encoding: str,
) -> Dict[str, set[str]]:
    tagged: Dict[str, set[str]] = {col: set() for col in challenge_cols}
    for path in tag_csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Tag CSV not found: {path}")
        with path.open("r", encoding=csv_encoding, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if text_id_col not in fieldnames:
                raise ValueError(f"Tag CSV missing '{text_id_col}': {path}")
            present = [col for col in challenge_cols if col in fieldnames]
            for row in reader:
                text_id = str(row.get(text_id_col, "")).strip()
                if not text_id:
                    continue
                for col in present:
                    if is_true(row.get(col)):
                        tagged[col].add(text_id)
    return tagged


def read_scored_rows(
    path: Path,
    text_id_col: str,
    match_col: str,
    jaso_score_col: str,
    csv_encoding: str,
) -> List[tuple[str, float, float]]:
    with path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        missing = {text_id_col, match_col, jaso_score_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Scored CSV missing columns {sorted(missing)}: {path}")
        return [
            (
                str(row.get(text_id_col, "")).strip(),
                float(row[match_col]),
                float(row[jaso_score_col]),
            )
            for row in reader
        ]


def summarize(rows: Iterable[tuple[str, float, float]]) -> tuple[int, float, float]:
    count = 0
    match_sum = 0.0
    jaso_sum = 0.0
    for _text_id, is_match, jaso_score in rows:
        count += 1
        match_sum += is_match
        jaso_sum += jaso_score
    if count == 0:
        return 0, 0.0, 0.0
    return count, match_sum / count, jaso_sum / count


def execute(
    scored_csv_dir: Path,
    output_csv_path: Path,
    regimes: Sequence[str],
    models: Sequence[str],
    dataset_sources: Mapping[str, Sequence[str]],
    challenge_cols: Sequence[str] = (),
    tag_csv_by_source: Mapping[str, Path] | None = None,
    text_id_col: str = "text_id",
    match_col: str = "is_match",
    jaso_score_col: str = "jaso_1_ned",
    skip_missing: bool = True,
    skip_empty_challenge: bool = True,
    csv_encoding: str = DEFAULT_CSV_ENCODING,
) -> Dict[str, Any]:
    tag_csv_by_source = dict(tag_csv_by_source or {})
    needed_sources = {source for sources in dataset_sources.values() for source in sources}
    if challenge_cols and not tag_csv_by_source:
        raise ValueError("tag_csv_by_source is required when challenge_cols is given")
    missing_sources = needed_sources - set(tag_csv_by_source)
    if challenge_cols and missing_sources:
        raise ValueError(f"tag_csv_by_source is missing sources: {sorted(missing_sources)}")

    tagged_by_source = {
        source: load_tagged_ids([path], challenge_cols, text_id_col, csv_encoding)
        for source, path in tag_csv_by_source.items()
        if source in needed_sources
    }
    tagged_by_dataset: Dict[str, Dict[str, set[str]]] = {}
    for dataset, sources in dataset_sources.items():
        merged = {col: set() for col in challenge_cols}
        for source in sources:
            for col, ids in tagged_by_source.get(source, {}).items():
                merged[col].update(ids)
        tagged_by_dataset[dataset] = merged

    out_rows: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for regime in regimes:
        for model in models:
            for dataset, sources in dataset_sources.items():
                rows: List[tuple[str, float, float]] = []
                found = False
                for source in sources:
                    path = scored_csv_dir / regime / model / f"{source}.csv"
                    if not path.exists():
                        if not skip_missing:
                            raise FileNotFoundError(f"Scored CSV not found: {path}")
                        missing_files.append(str(path))
                        continue
                    found = True
                    rows.extend(
                        read_scored_rows(
                            path, text_id_col, match_col, jaso_score_col, csv_encoding
                        )
                    )
                if not found:
                    continue

                for challenge in ("", *challenge_cols):
                    if challenge:
                        ids = tagged_by_dataset[dataset][challenge]
                        subset = [row for row in rows if row[0] in ids]
                    else:
                        subset = rows
                    count, accuracy, jaso_1_ned = summarize(subset)
                    if challenge and skip_empty_challenge and count == 0:
                        continue
                    out_rows.append(
                        {
                            "regime": regime,
                            "model": model,
                            "dataset": dataset,
                            "challenge": challenge,
                            "samples": count,
                            "accuracy": accuracy,
                            "jaso_1_ned": jaso_1_ned,
                        }
                    )

    fieldnames = [
        "regime",
        "model",
        "dataset",
        "challenge",
        "samples",
        "accuracy",
        "jaso_1_ned",
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding=csv_encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"scored_csv_dir  : {scored_csv_dir}")
    print(f"output_csv_path : {output_csv_path}")
    print(f"regimes         : {list(regimes)}")
    print(f"models          : {len(models)}")
    print(f"datasets        : {list(dataset_sources)}")
    print(f"challenges      : {list(challenge_cols)}")
    print(f"summary rows    : {len(out_rows)}")
    print(f"missing files   : {len(missing_files)}")

    return {
        "scored_csv_dir": str(scored_csv_dir),
        "output_csv_path": str(output_csv_path),
        "regimes": list(regimes),
        "models": list(models),
        "datasets": list(dataset_sources),
        "challenges": list(challenge_cols),
        "summary_rows": len(out_rows),
        "missing_files": missing_files,
        "csv_encoding": csv_encoding,
    }


@dataclass(kw_only=True)
class SummarizeInferScoresStep(Step):
    scored_csv_dir: str
    output_csv_path: str
    regimes: List[str]
    models: List[str]
    dataset_sources: Dict[str, List[str]]
    challenge_cols: List[str] = field(default_factory=list)
    tag_csv_by_source: Dict[str, str] = field(default_factory=dict)
    text_id_col: str = "text_id"
    match_col: str = "is_match"
    jaso_score_col: str = "jaso_1_ned"
    skip_missing: bool = True
    skip_empty_challenge: bool = True
    csv_encoding: str = DEFAULT_CSV_ENCODING

    def run(self) -> Dict[str, Any]:
        return execute(
            scored_csv_dir=Path(self.scored_csv_dir),
            output_csv_path=Path(self.output_csv_path),
            regimes=self.regimes,
            models=self.models,
            dataset_sources=self.dataset_sources,
            challenge_cols=self.challenge_cols,
            tag_csv_by_source={source: Path(path) for source, path in self.tag_csv_by_source.items()},
            text_id_col=self.text_id_col,
            match_col=self.match_col,
            jaso_score_col=self.jaso_score_col,
            skip_missing=self.skip_missing,
            skip_empty_challenge=self.skip_empty_challenge,
            csv_encoding=self.csv_encoding,
        )
