from __future__ import annotations

import csv
import json
import random
import unicodedata
from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [str(value)]


def _parse_ratio(raw: Any) -> tuple[float, float, float]:
    parts = _as_list(raw)
    if not parts:
        return 0.8, 0.1, 0.1
    if len(parts) != 3:
        raise ValueError("train_val_test_ratio must have exactly 3 values")
    vals = [float(x) for x in parts]
    if any(v < 0 for v in vals):
        raise ValueError("train_val_test_ratio values must be >= 0")
    total = sum(vals)
    if total <= 0:
        raise ValueError("sum(train_val_test_ratio) must be > 0")
    return vals[0] / total, vals[1] / total, vals[2] / total


def _match_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return any(k in text for k in keywords)


def _is_korean(ch: str) -> bool:
    code = ord(ch)
    # Treat only complete Hangul syllables as Korean.
    # Jamo blocks are intentionally excluded so they can be handled as "others".
    return 0xAC00 <= code <= 0xD7A3


def _is_english(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_special(ch: str) -> bool:
    if _is_korean(ch) or _is_english(ch) or ch.isdigit():
        return False
    if ch.isspace():
        return False
    return unicodedata.category(ch).startswith(("P", "S"))


def _has_other(ch: str) -> bool:
    return (not _is_korean(ch)) and (not _is_english(ch)) and (not ch.isdigit()) and (not _is_special(ch)) and (not ch.isspace())


def _coerce_int_for_sort(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_filters(raw: Any) -> list[dict[str, Any]]:
    parts = raw if isinstance(raw, (list, tuple)) else []
    parsed: list[dict[str, Any]] = []
    allowed_ops = {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "contains", "not_contains"}
    for i, item in enumerate(parts):
        if not isinstance(item, dict):
            raise ValueError(f"filters[{i}] must be an object")
        col = str(item.get("column", "")).strip()
        op = str(item.get("op", "in")).strip().lower()
        value = item.get("value")
        values = _as_list(item.get("values"))
        if not col:
            raise ValueError(f"filters[{i}].column is required")
        if op not in allowed_ops:
            raise ValueError(f"filters[{i}].op is invalid: {op}")
        if op in ("in", "not_in") and not values:
            raise ValueError(f"filters[{i}].values is required for op={op}")
        if op in ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains") and value is None:
            raise ValueError(f"filters[{i}].value is required for op={op}")
        exclude = coerce_bool(item.get("exclude", False), default=False)
        parsed.append({"column": col, "op": op, "value": value, "values": values, "exclude": exclude})
    return parsed


def _evaluate_filter(raw_value: Any, filt: dict[str, Any]) -> bool:
    op = str(filt.get("op", "in"))
    text = str(raw_value) if raw_value is not None else ""
    if op == "in":
        return text in {str(v) for v in filt.get("values", [])}
    if op == "not_in":
        return text not in {str(v) for v in filt.get("values", [])}
    if op == "eq":
        return text == str(filt.get("value"))
    if op == "ne":
        return text != str(filt.get("value"))
    if op == "contains":
        return str(filt.get("value")) in text
    if op == "not_contains":
        return str(filt.get("value")) not in text
    if op in ("gt", "gte", "lt", "lte"):
        left = _to_float(raw_value)
        right = _to_float(filt.get("value"))
        if left is None or right is None:
            return False
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right
    return False


class SplitAnnotationsCsvStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    in_csv_file_raw = params.get("in_csv_file")
    out_dir_path_raw = params.get("out_dir_path")
    out_config_file_raw = params.get("out_config_file")
    out_train_csv_file_raw = params.get("out_train_csv_file")
    out_val_csv_file_raw = params.get("out_val_csv_file")
    out_test_csv_file_raw = params.get("out_test_csv_file")
    make_ppocr_label = coerce_bool(params.get("make_ppocr_label", False), default=False)
    ppocr_image_col = str(params.get("ppocr_image_col", "image_path"))
    ppocr_text_col = str(params.get("ppocr_text_col", "text"))
    out_train_label_file_raw = params.get("out_train_label_file")
    out_val_label_file_raw = params.get("out_val_label_file")
    out_test_label_file_raw = params.get("out_test_label_file")

    use_shuffle = coerce_bool(params.get("use_shuffle", True), default=True)
    seed = int(params.get("seed", 42))
    ratio_train, ratio_val, ratio_test = _parse_ratio(
        params.get("train_val_test_ratio", [0.8, 0.1, 0.1])
    )
    include_keywords = _as_list(params.get("include_keywords"))
    exclude_keywords = _as_list(params.get("exclude_keywords"))
    keyword_columns = _as_list(params.get("keyword_columns", ["text"]))
    group_columns = _as_list(params.get("group_column", params.get("group_columns")))
    filters = _parse_filters(params.get("filters"))
    exclude_kor = coerce_bool(params.get("exclude_kor", False), default=False)
    exclude_eng = coerce_bool(params.get("exclude_eng", False), default=False)
    exclude_digit = coerce_bool(params.get("exclude_digit", False), default=False)
    exclude_spacial = coerce_bool(params.get("exclude_spacial", params.get("exclude_special", False)), default=False)
    exclude_space = coerce_bool(params.get("exclude_space", False), default=False)
    exclude_others = coerce_bool(params.get("exclude_others", False), default=False)
    sort_by_annotation_id = coerce_bool(params.get("sort_by_annotation_id", False), default=False)
    annotation_id_col = str(params.get("annotation_id_col", "annotation_id"))
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_train_csv_file_raw:
        raise ValueError("out_train_csv_file is required")
    if not out_val_csv_file_raw:
        raise ValueError("out_val_csv_file is required")
    if not out_test_csv_file_raw:
        raise ValueError("out_test_csv_file is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_dir_path = (
        Path(str(out_dir_path_raw)).expanduser().resolve() if out_dir_path_raw else None
    )

    def resolve_output_path(raw: Any) -> Path:
        p = Path(str(raw)).expanduser()
        if p.is_absolute():
            return p.resolve()
        if out_dir_path is not None:
            return (out_dir_path / p).resolve()
        return p.resolve()

    out_train_csv_file = resolve_output_path(out_train_csv_file_raw)
    out_val_csv_file = resolve_output_path(out_val_csv_file_raw)
    out_test_csv_file = resolve_output_path(out_test_csv_file_raw)
    out_config_file = (
        resolve_output_path(out_config_file_raw) if out_config_file_raw else None
    )
    out_train_label_file = (
        resolve_output_path(out_train_label_file_raw)
        if out_train_label_file_raw
        else out_train_csv_file.with_suffix(".txt")
    )
    out_val_label_file = (
        resolve_output_path(out_val_label_file_raw)
        if out_val_label_file_raw
        else out_val_csv_file.with_suffix(".txt")
    )
    out_test_label_file = (
        resolve_output_path(out_test_label_file_raw)
        if out_test_label_file_raw
        else out_test_csv_file.with_suffix(".txt")
    )
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("split_annotations_csv_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_dir_path": str(out_dir_path) if out_dir_path else None,
            "out_config_file": str(out_config_file) if out_config_file else None,
            "out_train_csv_file": str(out_train_csv_file),
            "out_val_csv_file": str(out_val_csv_file),
            "out_test_csv_file": str(out_test_csv_file),
            "make_ppocr_label": make_ppocr_label,
            "ppocr_image_col": ppocr_image_col,
            "ppocr_text_col": ppocr_text_col,
            "out_train_label_file": str(out_train_label_file),
            "out_val_label_file": str(out_val_label_file),
            "out_test_label_file": str(out_test_label_file),
            "use_shuffle": use_shuffle,
            "seed": seed,
            "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
            "include_keywords": include_keywords,
            "exclude_keywords": exclude_keywords,
            "keyword_columns": keyword_columns,
            "group_columns": group_columns,
            "filters": filters,
            "exclude_kor": exclude_kor,
            "exclude_eng": exclude_eng,
            "exclude_digit": exclude_digit,
            "exclude_spacial": exclude_spacial,
            "exclude_space": exclude_space,
            "exclude_others": exclude_others,
            "sort_by_annotation_id": sort_by_annotation_id,
            "annotation_id_col": annotation_id_col,
            "log_level": log_level,
        },
        title="split annotations discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        for f in filters:
            if str(f["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{f['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    # keyword 필터링
    filtered_rows: list[dict[str, str]] = []
    excluded_by_include = 0
    excluded_by_exclude = 0
    excluded_by_char_filter = 0
    excluded_by_filters = 0
    for row in rows:
        if filters:
            matched = True
            for f in filters:
                cond = _evaluate_filter(row.get(str(f["column"]), ""), f)
                if bool(f.get("exclude", False)):
                    cond = not cond
                if not cond:
                    matched = False
                    break
            if not matched:
                excluded_by_filters += 1
                continue
        target_text = " ".join(str(row.get(col, "") or "") for col in keyword_columns)
        if include_keywords and not _match_keywords(target_text, include_keywords):
            excluded_by_include += 1
            continue
        if exclude_keywords and _match_keywords(target_text, exclude_keywords):
            excluded_by_exclude += 1
            continue
        if exclude_kor and any(_is_korean(ch) for ch in target_text):
            excluded_by_char_filter += 1
            continue
        if exclude_eng and any(_is_english(ch) for ch in target_text):
            excluded_by_char_filter += 1
            continue
        if exclude_digit and any(ch.isdigit() for ch in target_text):
            excluded_by_char_filter += 1
            continue
        if exclude_spacial and any(_is_special(ch) for ch in target_text):
            excluded_by_char_filter += 1
            continue
        if exclude_space and any(ch.isspace() for ch in target_text):
            excluded_by_char_filter += 1
            continue
        if exclude_others and any(_has_other(ch) for ch in target_text):
            excluded_by_char_filter += 1
            continue
        filtered_rows.append(row)

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    if group_columns:
        for row in filtered_rows:
            key = tuple(str(row.get(col, "") or "") for col in group_columns)
            grouped.setdefault(key, []).append(row)
    else:
        grouped[("__all__",)] = filtered_rows

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    rng = random.Random(seed)

    ordered_keys = sorted(grouped.keys())
    for key in ordered_keys:
        group_rows = list(grouped[key])
        if use_shuffle:
            rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = int(n * ratio_train)
        n_val = int(n * ratio_val)
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        train_rows.extend(group_rows[:n_train])
        val_rows.extend(group_rows[n_train : n_train + n_val])
        test_rows.extend(group_rows[n_train + n_val : n_train + n_val + n_test])

    if sort_by_annotation_id:
        train_rows.sort(
            key=lambda r: _coerce_int_for_sort(r.get(annotation_id_col), default=0)
        )
        val_rows.sort(
            key=lambda r: _coerce_int_for_sort(r.get(annotation_id_col), default=0)
        )
        test_rows.sort(
            key=lambda r: _coerce_int_for_sort(r.get(annotation_id_col), default=0)
        )

    out_train_csv_file.parent.mkdir(parents=True, exist_ok=True)
    out_val_csv_file.parent.mkdir(parents=True, exist_ok=True)
    out_test_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_train_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train_rows)
    with out_val_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(val_rows)
    with out_test_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_rows)

    def write_ppocr_label_file(out_file: Path, rows_local: list[dict[str, str]]) -> int:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        line_count = 0
        with out_file.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows_local:
                img = str(row.get(ppocr_image_col, "") or "").strip()
                txt = str(row.get(ppocr_text_col, "") or "")
                if not img:
                    continue
                f.write(f"{img}\t{txt}\n")
                line_count += 1
        return line_count

    train_label_count = 0
    val_label_count = 0
    test_label_count = 0
    if make_ppocr_label:
        train_label_count = write_ppocr_label_file(out_train_label_file, train_rows)
        val_label_count = write_ppocr_label_file(out_val_label_file, val_rows)
        test_label_count = write_ppocr_label_file(out_test_label_file, test_rows)

    if out_config_file is not None:
        out_config_file.parent.mkdir(parents=True, exist_ok=True)
        effective_config = {
            "in_csv_file": str(in_csv_file),
            "out_dir_path": str(out_dir_path) if out_dir_path else None,
            "out_train_csv_file": str(out_train_csv_file),
            "out_val_csv_file": str(out_val_csv_file),
            "out_test_csv_file": str(out_test_csv_file),
            "use_shuffle": use_shuffle,
            "seed": seed,
            "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
            "include_keywords": include_keywords,
            "exclude_keywords": exclude_keywords,
            "filters": filters,
            "exclude_kor": exclude_kor,
            "exclude_eng": exclude_eng,
            "exclude_digit": exclude_digit,
            "exclude_spacial": exclude_spacial,
            "exclude_space": exclude_space,
            "exclude_others": exclude_others,
            "sort_by_annotation_id": sort_by_annotation_id,
            "annotation_id_col": annotation_id_col,
            "keyword_columns": keyword_columns,
            "group_columns": group_columns,
            "make_ppocr_label": make_ppocr_label,
            "ppocr_image_col": ppocr_image_col,
            "ppocr_text_col": ppocr_text_col,
            "out_train_label_file": str(out_train_label_file) if make_ppocr_label else None,
            "out_val_label_file": str(out_val_label_file) if make_ppocr_label else None,
            "out_test_label_file": str(out_test_label_file) if make_ppocr_label else None,
            "log_level": log_level,
        }
        with out_config_file.open("w", encoding="utf-8") as f:
            json.dump(effective_config, f, ensure_ascii=False, indent=2)

    result = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_dir_path": str(out_dir_path) if out_dir_path else None,
        "out_config_file": str(out_config_file) if out_config_file else None,
        "out_train_csv_file": str(out_train_csv_file),
        "out_val_csv_file": str(out_val_csv_file),
        "out_test_csv_file": str(out_test_csv_file),
        "make_ppocr_label": make_ppocr_label,
        "ppocr_image_col": ppocr_image_col,
        "ppocr_text_col": ppocr_text_col,
        "out_train_label_file": str(out_train_label_file) if make_ppocr_label else None,
        "out_val_label_file": str(out_val_label_file) if make_ppocr_label else None,
        "out_test_label_file": str(out_test_label_file) if make_ppocr_label else None,
        "train_label_count": train_label_count if make_ppocr_label else None,
        "val_label_count": val_label_count if make_ppocr_label else None,
        "test_label_count": test_label_count if make_ppocr_label else None,
        "use_shuffle": use_shuffle,
        "seed": seed,
        "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
        "input_count": len(rows),
        "filtered_count": len(filtered_rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "excluded_by_include_count": excluded_by_include,
        "excluded_by_exclude_count": excluded_by_exclude,
        "excluded_by_filters_count": excluded_by_filters,
        "excluded_by_char_filter_count": excluded_by_char_filter,
        "group_columns": group_columns,
        "group_count": len(grouped),
        "keyword_columns": keyword_columns,
        "include_keywords": include_keywords,
        "exclude_keywords": exclude_keywords,
        "filters": filters,
        "exclude_kor": exclude_kor,
        "exclude_eng": exclude_eng,
        "exclude_digit": exclude_digit,
        "exclude_spacial": exclude_spacial,
        "exclude_space": exclude_space,
        "exclude_others": exclude_others,
        "sort_by_annotation_id": sort_by_annotation_id,
        "annotation_id_col": annotation_id_col,
        "log_level": log_level,
    }
    logger.log(result, title="split annotations summary")
    emit(
        log_level,
        "INFO",
        (
            f"[split-ann] input={len(rows)}, filtered={len(filtered_rows)}, "
            f"train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}"
        ),
    )
    if make_ppocr_label:
        emit(
            log_level,
            "INFO",
            (
                f"[split-ann] ppocr labels train={train_label_count}, "
                f"val={val_label_count}, test={test_label_count}"
            ),
        )
    return result

