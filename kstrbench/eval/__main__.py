"""Score one inference file against a K-STR-Bench subset.

txt and jsonl inputs are converted to csv and saved first. The csv is then
joined with the subset labels, scored, and saved.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kstrbench.dataset_dir import AIHUB, GIST, KAIST, dataset_root, text_csv
from kstrbench.eval.steps.score_infer_csv import compute_metrics

CSV_ENCODING = "utf-8-sig"
TEXT_ID_COL = "text_id"
INFER_COL = "infer"
LABEL_COL = "label"
MATCH_COL = "is_match"
JASO_COL = "jaso_1_ned"

DATASETS = (AIHUB, GIST, KAIST)

# Printed name, then the label-CSV column. GIST and KAIST only.
CHALLENGES = (
    ("artistic", "KSTR_is_artistic"),
    ("curve", "KSTR_is_curve"),
    ("sailent", "KSTR_is_sailent"),
    ("incomplete", "KSTR_is_incomplete"),
    ("multi_oriented", "KSTR_is_multi_oriented"),
    ("low_visibility", "KSTR_is_low_visibility"),
)
CHALLENGE_DATASETS = {GIST, KAIST}


def text_id_from_image_path(path_text: str) -> str:
    normalized = str(path_text or "").strip().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return Path(name).stem


def read_txt_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n\r")
            if not raw.strip():
                continue
            if "\t" not in raw:
                raise ValueError(f"{path}:{line_no}: expected image_path<TAB>infer")
            image_path, infer = raw.split("\t", 1)
            pairs.append((text_id_from_image_path(image_path), infer.strip()))
    return pairs


def read_jsonl_pairs(path: Path, path_key: str, infer_key: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: jsonl row must be an object")
            missing = [key for key in (path_key, infer_key) if key not in item]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {missing}")
            image_path = "" if item[path_key] is None else str(item[path_key])
            infer = "" if item[infer_key] is None else str(item[infer_key])
            pairs.append((text_id_from_image_path(image_path), infer.strip()))
    return pairs


def write_pred_csv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[TEXT_ID_COL, INFER_COL],
            lineterminator="\n",
        )
        writer.writeheader()
        for text_id, infer in pairs:
            writer.writerow({TEXT_ID_COL: text_id, INFER_COL: infer})


def read_pred_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        missing = {TEXT_ID_COL, INFER_COL} - set(fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing columns {sorted(missing)}: {path}. Available: {fieldnames}"
            )
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "t"}


def load_label_map(dataset: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    label_csv = text_csv(dataset_root(), dataset)
    if not label_csv.is_file():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    challenge_cols = CHALLENGES if dataset in CHALLENGE_DATASETS else ()
    labels: dict[str, str] = {}
    tagged: dict[str, set[str]] = {name: set() for name, _col in challenge_cols}
    with label_csv.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if TEXT_ID_COL not in fieldnames:
            raise ValueError(f"Label CSV missing '{TEXT_ID_COL}': {label_csv}")
        if LABEL_COL not in fieldnames:
            raise ValueError(f"Label CSV missing '{LABEL_COL}': {label_csv}")
        missing = [col for _name, col in challenge_cols if col not in fieldnames]
        if missing:
            raise ValueError(f"Label CSV missing columns {missing}: {label_csv}")
        for row in reader:
            text_id = str(row.get(TEXT_ID_COL) or "").strip()
            if not text_id:
                continue
            labels[text_id] = str(row.get(LABEL_COL) or "")
            for name, col in challenge_cols:
                if is_true(row.get(col)):
                    tagged[name].add(text_id)
    return labels, tagged


def score_rows(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    labels: dict[str, str],
) -> tuple[list[str], int, int, float, float, list[tuple[str, int, float]]]:
    out_fields = list(fieldnames)
    for col in (LABEL_COL, MATCH_COL, JASO_COL):
        if col not in out_fields:
            out_fields.append(col)

    matched = 0
    match_sum = 0
    jaso_sum = 0.0
    infer_ids: set[str] = set()
    matched_scores: list[tuple[str, int, float]] = []
    for row in rows:
        text_id = str(row.get(TEXT_ID_COL) or "").strip()
        if text_id:
            infer_ids.add(text_id)
        label = labels.get(text_id)
        if label is None:
            row[LABEL_COL] = ""
            row[MATCH_COL] = ""
            row[JASO_COL] = ""
            continue
        is_match, jaso_score = compute_metrics(
            label_text=label,
            infer_text=str(row.get(INFER_COL) or ""),
            ignore_blank=True,
            is_lower=False,
            wildcard="□",
        )
        row[LABEL_COL] = label
        row[MATCH_COL] = is_match
        row[JASO_COL] = f"{jaso_score:.6f}"
        matched += 1
        match_sum += is_match
        jaso_sum += jaso_score
        matched_scores.append((text_id, is_match, jaso_score))

    unmatched_dataset = sum(1 for text_id in labels if text_id not in infer_ids)
    if matched == 0:
        accuracy = float("nan")
        jaso_1_ned = float("nan")
    else:
        accuracy = match_sum / matched
        jaso_1_ned = jaso_sum / matched
    return out_fields, matched, unmatched_dataset, accuracy, jaso_1_ned, matched_scores


def challenge_means(
    matched_scores: list[tuple[str, int, float]],
    tagged: dict[str, set[str]],
) -> list[tuple[str, int, float, float]]:
    reports: list[tuple[str, int, float, float]] = []
    for name, _col in CHALLENGES:
        chosen = [(is_match, jaso) for text_id, is_match, jaso in matched_scores if text_id in tagged[name]]
        count = len(chosen)
        if count == 0:
            reports.append((name, 0, float("nan"), float("nan")))
            continue
        accuracy = sum(is_match for is_match, _jaso in chosen) / count
        jaso_1_ned = sum(jaso for _is_match, jaso in chosen) / count
        reports.append((name, count, accuracy, jaso_1_ned))
    return reports


def format_challenge_line(name: str, count: int, accuracy: float, jaso_1_ned: float, sample_width: int) -> str:
    acc_text = "n/a" if count == 0 else f"{accuracy:.6f}"
    jaso_text = "n/a" if count == 0 else f"{jaso_1_ned:.6f}"
    return (
        f"  {name:<16} samples={count:>{sample_width}}  "
        f"ACC={acc_text:<8}  1-G_NED={jaso_text}"
    )


def write_scored_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def scored_csv_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.eval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one inference file against a K-STR-Bench subset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Subset name.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Inference file (.txt, .jsonl, or .csv). txt is image_path<TAB>infer. csv needs text_id and infer columns.",
    )
    parser.add_argument(
        "--path-key",
        default=None,
        help="JSONL key for the image path. Required for .jsonl.",
    )
    parser.add_argument(
        "--infer-key",
        default=None,
        help="JSONL key for the prediction. Required for .jsonl.",
    )
    args = parser.parse_args()
    pred_path = Path(args.pred)
    if pred_path.suffix.lower() == ".jsonl" and (not args.path_key or not args.infer_key):
        parser.error("--path-key and --infer-key are required for a .jsonl file")
    return args


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Inference file not found: {pred_path}")

    suffix = pred_path.suffix.lower()
    if suffix == ".txt":
        csv_path = pred_path.with_suffix(".csv")
        write_pred_csv(csv_path, read_txt_pairs(pred_path))
    elif suffix == ".jsonl":
        csv_path = pred_path.with_suffix(".csv")
        write_pred_csv(csv_path, read_jsonl_pairs(pred_path, args.path_key, args.infer_key))
    elif suffix == ".csv":
        csv_path = pred_path
    else:
        raise ValueError(f"Unsupported inference file type '{suffix}'. Use .txt, .jsonl, or .csv.")

    fieldnames, rows = read_pred_csv(csv_path)
    labels, tagged = load_label_map(args.dataset)
    out_fields, matched, unmatched_dataset, accuracy, jaso_1_ned, matched_scores = score_rows(
        fieldnames, rows, labels
    )
    out_path = scored_csv_path(csv_path)
    write_scored_csv(out_path, out_fields, rows)

    infer_count = len(rows)
    unmatched_infer = infer_count - matched
    print(f"dataset           : {args.dataset}")
    print(f"pred              : {pred_path}")
    print(f"csv               : {csv_path}")
    print(f"output            : {out_path}")
    print(f"inference results : {infer_count}")
    print(f"matched           : {matched}")
    print(f"unmatched infer   : {unmatched_infer}")
    print(f"unmatched dataset : {unmatched_dataset}")
    if matched == 0:
        print("accuracy          : n/a")
        print("jaso_1_ned        : n/a")
    else:
        print(f"accuracy          : {accuracy:.6f}")
        print(f"jaso_1_ned        : {jaso_1_ned:.6f}")
    if tagged:
        print("challenges")
        challenge_rows = challenge_means(matched_scores, tagged)
        sample_width = max(len(str(count)) for _name, count, _acc, _jaso in challenge_rows)
        for name, count, challenge_acc, challenge_jaso in challenge_rows:
            print(format_challenge_line(name, count, challenge_acc, challenge_jaso, sample_width))


if __name__ == "__main__":
    main()
