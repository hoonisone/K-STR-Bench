from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _extract_label(line: str) -> str | None:
    raw = line.rstrip("\r\n").strip()
    if not raw:
        return None
    if "\t" in raw:
        _, label = raw.split("\t", 1)
        return label.strip()

    parts = raw.split(maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count labels (excluding image path) in two files and print mismatched labels."
        )
    )
    parser.add_argument("file_a", type=Path, help="First file path")
    parser.add_argument("file_b", type=Path, help="Second file path")
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding (default: utf-8)",
    )
    return parser.parse_args()


def _count_labels(path: Path, encoding: str) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    total = 0
    skipped = 0
    with path.open("r", encoding=encoding) as f:
        for raw in f:
            total += 1
            label = _extract_label(raw)
            if label is None:
                skipped += 1
                continue
            counts[label] += 1
    return counts, total, skipped


def compare_label_counts(file_a: Path, file_b: Path, encoding: str = "utf-8") -> int:
    if not file_a.exists():
        raise FileNotFoundError(f"File not found: {file_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"File not found: {file_b}")

    a_counts, a_total, a_skipped = _count_labels(file_a, encoding)
    b_counts, b_total, b_skipped = _count_labels(file_b, encoding)

    print(f"A total lines: {a_total} (parsed labels: {a_total - a_skipped}, skipped: {a_skipped})")
    print(f"B total lines: {b_total} (parsed labels: {b_total - b_skipped}, skipped: {b_skipped})")

    all_labels = sorted(set(a_counts.keys()) | set(b_counts.keys()))
    diffs: list[tuple[str, int, int]] = []
    for label in all_labels:
        ca = a_counts.get(label, 0)
        cb = b_counts.get(label, 0)
        if ca != cb:
            diffs.append((label, ca, cb))

    if not diffs:
        print("All label counts match.")
        return 0

    print(f"Mismatched labels: {len(diffs)}")
    for label, ca, cb in diffs:
        print(f"{label}\tA={ca}\tB={cb}")
    return 1


def main() -> None:
    args = _parse_args()
    compare_label_counts(args.file_a, args.file_b, args.encoding)


if __name__ == "__main__":
    main()
