from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .split_annotations_csv_entry import (
    _as_list,
    _evaluate_filter,
    _has_other,
    _is_english,
    _is_korean,
    _is_special,
    _match_keywords,
    _parse_filters,
    _parse_ratio,
)
from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _split_paths(out_dir_path: Path | None, raw: Any) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    if out_dir_path is not None:
        return (out_dir_path / p).resolve()
    return p.resolve()


def _make_tags(row: dict[str, str], balance_columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in balance_columns:
        value = row.get(col)
        text = str(value).strip() if value is not None else ""
        out.append(f"{col}_{text if text else 'None'}")
    return out


class SplitAnnotationsTagBalancedGroupStep2:
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
    out_csv_file_raw = params.get("out_csv_file")
    split_col_name = str(params.get("split_col_name", "split")).strip()
    split_group_id_col = str(params.get("split_group_id_col", "image_id"))
    seed = int(params.get("seed", 42))
    ratio_train, ratio_val, ratio_test = _parse_ratio(params.get("train_val_test_ratio", [0.8, 0.1, 0.1]))
    include_keywords = _as_list(params.get("include_keywords"))
    exclude_keywords = _as_list(params.get("exclude_keywords"))
    keyword_columns = _as_list(params.get("keyword_columns", ["text"]))
    balance_columns = _as_list(params.get("group_column", params.get("group_columns")))
    filters = _parse_filters(params.get("filters"))
    exclude_kor = coerce_bool(params.get("exclude_kor", False), default=False)
    exclude_eng = coerce_bool(params.get("exclude_eng", False), default=False)
    exclude_digit = coerce_bool(params.get("exclude_digit", False), default=False)
    exclude_spacial = coerce_bool(params.get("exclude_spacial", params.get("exclude_special", False)), default=False)
    exclude_space = coerce_bool(params.get("exclude_space", False), default=False)
    exclude_others = coerce_bool(params.get("exclude_others", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file is required")
    if not split_group_id_col:
        raise ValueError("split_group_id_col is required")
    if not balance_columns:
        raise ValueError("group_column (or group_columns) is required")
    if not split_col_name:
        raise ValueError("split_col_name is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_dir_path = Path(str(out_dir_path_raw)).expanduser().resolve() if out_dir_path_raw else None
    out_csv_file = _split_paths(out_dir_path, out_csv_file_raw)
    out_config_file = _split_paths(out_dir_path, out_config_file_raw) if out_config_file_raw else None
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    logger = StepLogger("split_annotations_tag_balanced_group_entry2")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "out_config_file": str(out_config_file) if out_config_file else None,
            "split_group_id_col": split_group_id_col,
            "balance_columns": balance_columns,
            "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
            "filters": filters,
            "seed": seed,
            "show_progress": show_progress,
            "log_level": log_level,
        },
        title="split tag-balanced v2 discovery",
    )

    with in_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {in_csv_file}")
        if split_group_id_col not in fieldnames:
            raise ValueError(f"CSV missing split_group_id_col '{split_group_id_col}': {in_csv_file}")
        for col in balance_columns:
            if col not in fieldnames:
                raise ValueError(f"CSV missing balance column '{col}': {in_csv_file}")
        for flt in filters:
            if str(flt["column"]) not in fieldnames:
                raise ValueError(f"CSV missing filters.column '{flt['column']}': {in_csv_file}")
        rows = [dict(r) for r in reader]

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        if filters:
            matched = True
            for flt in filters:
                cond = _evaluate_filter(row.get(str(flt["column"]), ""), flt)
                if bool(flt.get("exclude", False)):
                    cond = not cond
                if not cond:
                    matched = False
                    break
            if not matched:
                continue
        target_text = " ".join(str(row.get(col, "") or "") for col in keyword_columns)
        if include_keywords and not _match_keywords(target_text, include_keywords):
            continue
        if exclude_keywords and _match_keywords(target_text, exclude_keywords):
            continue
        if exclude_kor and any(_is_korean(ch) for ch in target_text):
            continue
        if exclude_eng and any(_is_english(ch) for ch in target_text):
            continue
        if exclude_digit and any(ch.isdigit() for ch in target_text):
            continue
        if exclude_spacial and any(_is_special(ch) for ch in target_text):
            continue
        if exclude_space and any(ch.isspace() for ch in target_text):
            continue
        if exclude_others and any(_has_other(ch) for ch in target_text):
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError("no rows remain after filtering")

    image_rows: dict[str, list[dict[str, str]]] = {}
    image_tag_counts: dict[str, dict[str, int]] = {}
    tag_total_counts: dict[str, int] = defaultdict(int)
    tag_to_images: dict[str, set[str]] = defaultdict(set)
    for row in filtered_rows:
        gid = str(row.get(split_group_id_col, "") or "").strip() or "__missing_group__"
        image_rows.setdefault(gid, []).append(row)
        tags = _make_tags(row, balance_columns)
        image_tag_counts.setdefault(gid, {})
        for tag in tags:
            image_tag_counts[gid][tag] = int(image_tag_counts[gid].get(tag, 0)) + 1
            tag_total_counts[tag] = int(tag_total_counts.get(tag, 0)) + 1
            tag_to_images[tag].add(gid)

    target_ratios = [ratio_train, ratio_val, ratio_test]
    split_tag_counts: list[dict[str, int]] = [defaultdict(int), defaultdict(int), defaultdict(int)]  # type: ignore[assignment]
    split_totals = [0, 0, 0]
    tag_unassigned_images: dict[str, set[str]] = {tag: set(imgs) for tag, imgs in tag_to_images.items()}
    tag_unassigned_count: dict[str, int] = dict(tag_total_counts)
    unassigned_images: set[str] = set(image_rows.keys())
    assignments: dict[str, int] = {}

    def tag_dist(tag: str) -> float:
        total = max(1, int(tag_total_counts.get(tag, 0)))
        curr = [float(split_tag_counts[i].get(tag, 0)) / float(total) for i in range(3)]
        return sum(abs(curr[i] - target_ratios[i]) for i in range(3))

    def choose_split_for_image(gid: str, selected_tag: str) -> int:
        add = float(image_tag_counts.get(gid, {}).get(selected_tag, 0))
        total = max(1, int(tag_total_counts.get(selected_tag, 0)))
        # choose split whose ratio is most under target for selected tag
        deficits: list[float] = []
        for i in range(3):
            curr = float(split_tag_counts[i].get(selected_tag, 0)) / float(total)
            deficits.append(target_ratios[i] - curr)
        best = max(range(3), key=lambda i: (deficits[i], target_ratios[i]))
        # if no positive deficit, still pick largest target split
        if deficits[best] <= 0:
            best = max(range(3), key=lambda i: target_ratios[i])
        _ = add
        return int(best)

    pbar: Any = tqdm(total=len(unassigned_images), desc="tag-balanced v2 assignment", unit="image", dynamic_ncols=True) if show_progress else None
    while unassigned_images:
        candidates: list[tuple[str, int, float]] = []
        for tag in tag_total_counts.keys():
            remain = int(tag_unassigned_count.get(tag, 0))
            if remain <= 0:
                continue
            candidates.append((tag, remain, tag_dist(tag)))
        if not candidates:
            break
        # priority: remaining count asc -> distance desc -> tag name
        candidates.sort(key=lambda x: (x[1], -x[2], x[0]))
        selected_tag = candidates[0][0]
        cand_images = tag_unassigned_images.get(selected_tag, set())
        gid = next(iter(cand_images)) if cand_images else next(iter(unassigned_images))

        split_idx = choose_split_for_image(gid, selected_tag)
        assignments[gid] = split_idx
        split_totals[split_idx] += len(image_rows[gid])
        for tag, cnt in image_tag_counts.get(gid, {}).items():
            split_tag_counts[split_idx][tag] = int(split_tag_counts[split_idx].get(tag, 0)) + int(cnt)
            if gid in tag_unassigned_images.get(tag, set()):
                tag_unassigned_images[tag].remove(gid)
                tag_unassigned_count[tag] = int(tag_unassigned_count.get(tag, 0)) - int(cnt)
        unassigned_images.remove(gid)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    split_name_by_gid = {
        gid: ("train" if int(assignments.get(gid, 0)) == 0 else ("val" if int(assignments.get(gid, 0)) == 1 else "test"))
        for gid in image_rows.keys()
    }
    split_rows: list[dict[str, str]] = []
    for row in filtered_rows:
        gid = str(row.get(split_group_id_col, "") or "").strip() or "__missing_group__"
        out_row = dict(row)
        out_row[split_col_name] = split_name_by_gid.get(gid, "train")
        split_rows.append(out_row)
    out_fields = list(fieldnames)
    if split_col_name not in out_fields:
        out_fields.append(split_col_name)
    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(split_rows)

    if out_config_file is not None:
        out_config_file.parent.mkdir(parents=True, exist_ok=True)
        with out_config_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "in_csv_file": str(in_csv_file),
                    "out_csv_file": str(out_csv_file),
                    "split_col_name": split_col_name,
                    "split_group_id_col": split_group_id_col,
                    "balance_columns": balance_columns,
                    "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
                    "seed": seed,
                    "show_progress": show_progress,
                    "log_level": log_level,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    result = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "split_col_name": split_col_name,
        "split_group_id_col": split_group_id_col,
        "balance_columns": balance_columns,
        "train_val_test_ratio": [ratio_train, ratio_val, ratio_test],
        "input_count": len(rows),
        "filtered_count": len(filtered_rows),
        "group_id_count": len(image_rows),
        "tag_count": len(tag_total_counts),
        "train_count": sum(1 for r in split_rows if str(r.get(split_col_name, "")) == "train"),
        "val_count": sum(1 for r in split_rows if str(r.get(split_col_name, "")) == "val"),
        "test_count": sum(1 for r in split_rows if str(r.get(split_col_name, "")) == "test"),
        "show_progress": show_progress,
        "log_level": log_level,
    }
    logger.log(result, title="split tag-balanced v2 summary")
    emit(
        log_level,
        "INFO",
        (
            f"[split-tag-balance-v2] input={len(rows)}, filtered={len(filtered_rows)}, groups={len(image_rows)}, "
            f"tags={len(tag_total_counts)}"
        ),
    )
    return result
