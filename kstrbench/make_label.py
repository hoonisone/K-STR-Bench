from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kstrbench.dataset_dir import (
    AIHUB,
    CHALLENGE_FILES,
    GIST,
    KAIST,
    challenge_txt,
    dataset_root,
    split_txt,
    text_csv,
)

CSV_ENCODING = "utf-8-sig"
TXT_ENCODING = "utf-8"
SPLITS = ("train", "val", "test")


def is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "t"}


def is_false(value: object) -> bool:
    return str(value or "").strip().lower() in {"false", "0", "no", "f"}


def keep_gist_kaist_row(row: dict[str, str]) -> bool:
    return is_true(row.get("is_only_korean")) and is_false(row.get("is_illegible"))


def norm_path(path_text: str) -> str:
    return str(path_text or "").strip().replace("\\", "/")


def subset_image_path(row: dict[str, str]) -> str:
    rel = norm_path(row.get("image_path") or "").lstrip("/")
    marker = "text_images/"
    index = rel.find(marker)
    if index >= 0:
        return rel[index + len(marker) :]
    return rel


_ID_PREFIXES = ("aihub_", "gist_", "kaist_")


def jsonl_filename(image_path: str) -> str:
    rel = norm_path(image_path)
    folder, sep, name = rel.rpartition("/")
    for prefix in _ID_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return f"{folder}/{name}" if sep else name


def points_size(raw: str) -> tuple[int, int] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        if all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in value[:4]):
            xs = [float(item[0]) for item in value[:4]]
            ys = [float(item[1]) for item in value[:4]]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
        else:
            x, y = float(value[0]), float(value[1])
            w, h = float(value[2]), float(value[3])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    width = int(round(x + w)) - int(round(x))
    height = int(round(y + h)) - int(round(y))
    if width <= 0 or height <= 0:
        return None
    return width, height


def split_name(row: dict[str, str]) -> str:
    return str(row.get("split") or "").strip().lower()


def format_line(image_path: str, label: str) -> str:
    return f"{norm_path(image_path)}\t{label}\n"


def format_jsonl(image_path: str, label: str, width: int, height: int) -> str:
    row = {
        "filename": jsonl_filename(image_path),
        "text": label,
        "width": width,
        "height": height,
    }
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_lines(root: Path, path: Path, lines: list[str]) -> None:
    rel = path.relative_to(root)
    if not lines:
        print(f"{rel}: skip (empty)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding=TXT_ENCODING)
    print(f"{rel}: {len(lines)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def make_labels(root: Path, subset: str, image_path_of, extra_keep=None) -> None:
    by_split: dict[str, list[str]] = {name: [] for name in SPLITS}
    by_split_jsonl: dict[str, list[str]] = {name: [] for name in SPLITS}
    by_tag: dict[str, list[str]] = {out_name: [] for _, out_name in CHALLENGE_FILES}
    by_tag_jsonl: dict[str, list[str]] = {out_name: [] for _, out_name in CHALLENGE_FILES}

    for row in read_csv(text_csv(root, subset)):
        split = split_name(row)
        if split not in by_split:
            continue
        if extra_keep is not None and not extra_keep(row):
            continue
        label = row.get("label") or ""
        image_path = image_path_of(row)
        if not image_path or not str(label).strip():
            continue
        line = format_line(image_path, label)
        size = points_size(row.get("points") or "")
        jsonl_line = format_jsonl(image_path, label, *size) if size else ""
        by_split[split].append(line)
        if jsonl_line:
            by_split_jsonl[split].append(jsonl_line)
        for col, out_name in CHALLENGE_FILES:
            if is_true(row.get(col)):
                by_tag[out_name].append(line)
                if jsonl_line:
                    by_tag_jsonl[out_name].append(jsonl_line)

    for name in SPLITS:
        write_lines(root, split_txt(root, subset, name), by_split[name])
        write_lines(root, split_txt(root, subset, name).with_suffix(".jsonl"), by_split_jsonl[name])
    for out_name, lines in by_tag.items():
        write_lines(root, challenge_txt(root, subset, out_name), lines)
        write_lines(
            root,
            challenge_txt(root, subset, out_name).with_suffix(".jsonl"),
            by_tag_jsonl[out_name],
        )


def parse_args() -> str | None:
    parser = argparse.ArgumentParser(
        description="Generate split and challenge label files for K-STR-Bench"
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dataset root folder. If omitted, use the DATASET_DIR environment variable.",
    )
    return parser.parse_args().dataset_dir


def main() -> None:
    root = dataset_root(parse_args())
    make_labels(root, AIHUB, subset_image_path)
    make_labels(root, GIST, subset_image_path, extra_keep=keep_gist_kaist_row)
    make_labels(root, KAIST, subset_image_path, extra_keep=keep_gist_kaist_row)


if __name__ == "__main__":
    main()
