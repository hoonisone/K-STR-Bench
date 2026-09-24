from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [str(value).strip()]


def _safe_filename(text: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in text.strip())
    return out or "column"


def _is_korean(ch: str) -> bool:
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
    )


def _is_english(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_special(ch: str) -> bool:
    if _is_korean(ch) or _is_english(ch) or ch.isdigit():
        return False
    if ch.isspace():
        return False
    return unicodedata.category(ch).startswith(("P", "S"))


def _char_category(ch: str) -> str:
    if ch.isspace():
        return "space"
    if _is_korean(ch):
        return "korean"
    if _is_english(ch):
        return "english"
    if ch.isdigit():
        return "digit"
    if _is_special(ch):
        return "special"
    return "other"


def _classify_word(word: str) -> str:
    has_korean = False
    has_english = False
    has_plus = False
    for ch in word:
        if _is_korean(ch):
            has_korean = True
        elif _is_english(ch):
            has_english = True
        elif ch.isdigit() or _is_special(ch) or ch.isspace():
            has_plus = True
        else:
            has_plus = True
    if has_korean and not has_english and not has_plus:
        return "korean"
    if has_korean and not has_english and has_plus:
        return "korean_plus"
    if has_english and not has_korean and not has_plus:
        return "english"
    if has_english and not has_korean and has_plus:
        return "english_plus"
    if has_korean and has_english and not has_plus:
        return "korean_english"
    if has_korean and has_english and has_plus:
        return "korean_english_plus"
    return "others"


def _promote_group_with_space(group_name: str) -> str:
    if group_name == "korean":
        return "korean_plus"
    if group_name == "english":
        return "english_plus"
    if group_name == "korean_english":
        return "korean_english_plus"
    return group_name


class AnalyzeLabelTextStatisticsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file")
    out_dir_raw = params.get("out_dir")
    label_column = str(params.get("label_column", "text"))
    columns = _as_list(params.get("columns"))
    out_char_file_raw = params.get("out_char_file")
    out_char_category_summary_file_raw = params.get("out_char_category_summary_file")
    out_word_file_raw = params.get("out_word_file")
    out_word_group_summary_file_raw = params.get("out_word_group_summary_file")
    out_word_length_distribution_file_raw = params.get("out_word_length_distribution_file")
    make_word_distribution = coerce_bool(params.get("make_word_distribution", True), default=True)
    make_word_group_summary = coerce_bool(params.get("make_word_group_summary", True), default=True)
    make_word_length_distribution = coerce_bool(
        params.get("make_word_length_distribution", True),
        default=True,
    )
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    sort_by_count = coerce_bool(params.get("sort_by_count", True), default=True)
    lowercase_english = coerce_bool(params.get("lowercase_english", False), default=False)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_char_file_raw:
        raise ValueError("out_char_file is required")
    if make_word_distribution and not out_word_file_raw:
        raise ValueError("out_word_file is required")
    if make_word_group_summary and not out_word_file_raw and not out_word_group_summary_file_raw:
        raise ValueError("out_word_file or out_word_group_summary_file is required when make_word_group_summary=true")
    if make_word_length_distribution and not out_word_file_raw and not out_word_length_distribution_file_raw:
        raise ValueError("out_word_file or out_word_length_distribution_file is required when make_word_length_distribution=true")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_dir = Path(str(out_dir_raw)).expanduser().resolve() if out_dir_raw else None

    def resolve_output_path(raw: Any) -> Path:
        p = Path(str(raw)).expanduser()
        if p.is_absolute():
            return p.resolve()
        if out_dir is not None:
            return (out_dir / p).resolve()
        return p.resolve()

    out_char_file = resolve_output_path(out_char_file_raw)
    out_char_category_summary_file = (
        resolve_output_path(out_char_category_summary_file_raw)
        if out_char_category_summary_file_raw
        else out_char_file.with_name(f"{out_char_file.stem}_by_category{out_char_file.suffix}")
    )
    out_word_file = resolve_output_path(out_word_file_raw) if out_word_file_raw else None
    out_word_group_summary_file = (
        resolve_output_path(out_word_group_summary_file_raw)
        if out_word_group_summary_file_raw
        else (
            out_word_file.with_name(f"{out_word_file.stem}_by_group{out_word_file.suffix}")
            if out_word_file is not None
            else None
        )
    )
    out_word_length_distribution_file = (
        resolve_output_path(out_word_length_distribution_file_raw)
        if out_word_length_distribution_file_raw
        else (
            out_word_file.with_name(f"{out_word_file.stem}_length_distribution{out_word_file.suffix}")
            if out_word_file is not None
            else None
        )
    )
    if make_word_length_distribution and out_word_length_distribution_file is None:
        raise ValueError("out_word_length_distribution_file is required when make_word_length_distribution=true and out_word_file is omitted")
    if make_word_group_summary and out_word_group_summary_file is None:
        raise ValueError("out_word_group_summary_file is required when make_word_group_summary=true")
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("analyze_label_text_statistics_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_dir": str(out_dir) if out_dir else None,
            "label_column": label_column,
            "columns": columns,
            "out_char_file": str(out_char_file),
            "out_char_category_summary_file": str(out_char_category_summary_file),
            "out_word_file": str(out_word_file) if out_word_file else None,
            "out_word_group_summary_file": str(out_word_group_summary_file) if out_word_group_summary_file else None,
            "out_word_length_distribution_file": (
                str(out_word_length_distribution_file) if out_word_length_distribution_file else None
            ),
            "make_word_distribution": make_word_distribution,
            "make_word_group_summary": make_word_group_summary,
            "make_word_length_distribution": make_word_length_distribution,
            "show_progress": show_progress,
            "sort_by_count": sort_by_count,
            "lowercase_english": lowercase_english,
            "log_level": log_level,
        },
        title="label text statistics discovery",
    )

    char_counters: dict[str, Counter[str]] = {
        "korean": Counter(),
        "english": Counter(),
        "digit": Counter(),
        "special": Counter(),
        "space": Counter(),
        "other": Counter(),
    }
    word_group_counter: dict[str, Counter[str]] = {
        "korean": Counter(),
        "english": Counter(),
        "korean_english": Counter(),
        "others": Counter(),
        "korean_plus": Counter(),
        "english_plus": Counter(),
        "korean_english_plus": Counter(),
    }
    word_length_counter: Counter[int] = Counter()

    row_count = 0
    empty_label_count = 0
    total_char_count = 0
    total_word_count = 0
    column_value_counters: dict[str, Counter[str]] = {col: Counter() for col in columns}

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if label_column not in fields:
            raise ValueError(f"CSV missing label_column '{label_column}': {in_csv_file}")
        missing_columns = [col for col in columns if col not in fields]
        if missing_columns:
            raise ValueError(f"CSV missing columns: {missing_columns}")

        iter_rows: Any = (
            tqdm(reader, desc="analyze label text stats", unit="row", dynamic_ncols=True)
            if show_progress
            else reader
        )
        for row in iter_rows:
            row_count += 1
            text = str(row.get(label_column, "") or "")
            if text == "":
                empty_label_count += 1
            for col in columns:
                raw = row.get(col, "")
                val = str(raw) if raw is not None else ""
                if val == "":
                    val = "<EMPTY>"
                column_value_counters[col][val] += 1
            if text == "":
                continue

            for ch in text:
                cat = _char_category(ch)
                char_counters[cat][ch] += 1
                total_char_count += 1

            words = [w for w in text.split() if w]
            has_space_in_text = any(ch.isspace() for ch in text)
            for w in words:
                word = w.lower() if lowercase_english else w
                grp = _classify_word(word)
                if has_space_in_text:
                    grp = _promote_group_with_space(grp)
                word_group_counter[grp][word] += 1
                word_length_counter[len(word)] += 1
                total_word_count += 1

    # char detail + summary
    char_categories = ["korean", "english", "digit", "special", "space", "other"]
    char_rows: list[tuple[str, str, int]] = []
    for cat in char_categories:
        items = list(char_counters[cat].items())
        if sort_by_count:
            items.sort(key=lambda x: (-x[1], x[0]))
        else:
            items.sort(key=lambda x: x[0])
        char_rows.extend((cat, ch, cnt) for ch, cnt in items)

    out_char_file.parent.mkdir(parents=True, exist_ok=True)
    with out_char_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "character", "count"])
        for cat, ch, cnt in char_rows:
            writer.writerow([cat, ch, int(cnt)])

    out_char_category_summary_file.parent.mkdir(parents=True, exist_ok=True)
    with out_char_category_summary_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total_count"])
        for cat in char_categories:
            writer.writerow([cat, int(sum(char_counters[cat].values()))])

    # word group detail + summary
    word_rows: list[tuple[str, str, int]] = []
    for grp, counter in word_group_counter.items():
        items = list(counter.items())
        if sort_by_count:
            items.sort(key=lambda x: (-x[1], x[0]))
        else:
            items.sort(key=lambda x: x[0])
        word_rows.extend((grp, word, cnt) for word, cnt in items)

    if make_word_distribution:
        if out_word_file is None:
            raise ValueError("out_word_file is required when make_word_distribution=true")
        out_word_file.parent.mkdir(parents=True, exist_ok=True)
        with out_word_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "word", "count"])
            for grp, word, cnt in word_rows:
                writer.writerow([grp, word, int(cnt)])

    if make_word_group_summary:
        out_word_group_summary_file.parent.mkdir(parents=True, exist_ok=True)
        with out_word_group_summary_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "total_count"])
            for grp, counter in word_group_counter.items():
                writer.writerow([grp, int(sum(counter.values()))])

    # word length distribution
    if make_word_length_distribution:
        if out_word_length_distribution_file is None:
            raise ValueError("out_word_length_distribution_file is required when make_word_length_distribution=true")
        out_word_length_distribution_file.parent.mkdir(parents=True, exist_ok=True)
        with out_word_length_distribution_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word_length", "word_count"])
            for length in sorted(word_length_counter.keys()):
                writer.writerow([int(length), int(word_length_counter[length])])

    # selected columns value statistics
    column_stat_files: dict[str, str] = {}
    base_out_dir = out_dir if out_dir is not None else out_char_file.parent
    for col in columns:
        col_file = (base_out_dir / f"{_safe_filename(col)}.csv").resolve()
        items = list(column_value_counters[col].items())
        items.sort(key=lambda x: (-x[1], x[0]))
        col_file.parent.mkdir(parents=True, exist_ok=True)
        with col_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value", "count"])
            for value, cnt in items:
                writer.writerow([value, int(cnt)])
        column_stat_files[col] = str(col_file)

    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_dir": str(out_dir) if out_dir else None,
        "label_column": label_column,
        "out_char_file": str(out_char_file),
        "out_char_category_summary_file": str(out_char_category_summary_file),
        "out_word_file": str(out_word_file) if make_word_distribution and out_word_file else None,
        "out_word_group_summary_file": (
            str(out_word_group_summary_file) if make_word_group_summary else None
        ),
        "out_word_length_distribution_file": (
            str(out_word_length_distribution_file)
            if make_word_length_distribution and out_word_length_distribution_file
            else None
        ),
        "columns": columns,
        "column_stat_files": column_stat_files,
        "row_count": row_count,
        "empty_label_count": empty_label_count,
        "total_char_count": total_char_count,
        "total_word_count": total_word_count,
        "word_length_bucket_count": len(word_length_counter),
        "show_progress": show_progress,
        "sort_by_count": sort_by_count,
        "lowercase_english": lowercase_english,
        "make_word_distribution": make_word_distribution,
        "make_word_group_summary": make_word_group_summary,
        "make_word_length_distribution": make_word_length_distribution,
        "log_level": log_level,
    }
    logger.log(summary, title="label text statistics summary")
    emit(log_level, "INFO", f"[label-text-stats] in_csv={in_csv_file}")
    emit(log_level, "INFO", f"[label-text-stats] out_char={out_char_file}")
    if make_word_distribution and out_word_file is not None:
        emit(log_level, "INFO", f"[label-text-stats] out_word={out_word_file}")
    emit(
        log_level,
        "INFO",
        (
            f"[label-text-stats] rows={row_count}, total_chars={total_char_count}, "
            f"total_words={total_word_count}, length_buckets={len(word_length_counter)}"
        ),
    )
    return summary

