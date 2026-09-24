"""
이미지/라벨 경로 리스트를 키워드 매핑 기반으로 비교해 매칭 여부를 점검.

- image_list_file의 각 경로에 keyword_mapping(key -> value) 일괄 적용
- 적용된 image 경로 집합과 label 경로 집합을 비교
- 매칭 실패 경로 목록은 metadata 파일에 저장
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def run() -> None:
    run_from_config({})


def _normalize_path_text(text: str) -> str:
    return text.strip().replace("\\", "/")


def _apply_keyword_mapping(path_text: str, mapping: dict[str, str]) -> str:
    out = path_text
    for src, dst in mapping.items():
        out = out.replace(src, dst)
    return out


def _normalize_keyword_mapping(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("keyword_mapping must be a mapping")
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k)
        value = str(v)
        if not key:
            continue
        out[key] = value
    return out


class CheckMappedPathListMatchStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    image_list_file_raw = params.get("image_list_file")
    label_list_file_raw = params.get("label_list_file")
    out_unmatch_file_raw = params.get("out_unmatch_file")
    out_match_pair_file_raw = params.get("out_match_pair_file")
    keyword_mapping = _normalize_keyword_mapping(params.get("keyword_mapping", {}))
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    sample_limit = int(params.get("sample_limit", 50))
    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")

    if not image_list_file_raw:
        raise ValueError("image_list_file is required")
    if not label_list_file_raw:
        raise ValueError("label_list_file is required")
    if not out_unmatch_file_raw:
        raise ValueError("out_unmatch_file is required")

    image_list_file = Path(str(image_list_file_raw)).expanduser().resolve()
    label_list_file = Path(str(label_list_file_raw)).expanduser().resolve()
    out_unmatch_file = Path(str(out_unmatch_file_raw)).expanduser().resolve()
    out_match_pair_file = (
        Path(str(out_match_pair_file_raw)).expanduser().resolve()
        if out_match_pair_file_raw is not None
        else None
    )
    if not image_list_file.is_file():
        raise FileNotFoundError(f"image_list_file not found: {image_list_file}")
    if not label_list_file.is_file():
        raise FileNotFoundError(f"label_list_file not found: {label_list_file}")

    log_level = normalize_log_level(params.get("log_level"), default="INFO")
    logger = StepLogger("check_mapped_path_list_match_entry")

    with image_list_file.open("r", encoding="utf-8") as f:
        image_rows = [line.rstrip("\n") for line in f]
    with label_list_file.open("r", encoding="utf-8") as f:
        label_rows = [line.rstrip("\n") for line in f]

    image_iter: Any = (
        tqdm(image_rows, desc="map image paths", unit="line", dynamic_ncols=True)
        if show_progress
        else image_rows
    )
    mapped_image_rows: list[str] = []
    mapped_to_image_rows: dict[str, list[str]] = {}
    for row in image_iter:
        norm = _normalize_path_text(row)
        if not norm:
            continue
        mapped = _apply_keyword_mapping(norm, keyword_mapping)
        mapped_image_rows.append(mapped)
        mapped_to_image_rows.setdefault(mapped, []).append(norm)

    label_iter: Any = (
        tqdm(label_rows, desc="normalize label paths", unit="line", dynamic_ncols=True)
        if show_progress
        else label_rows
    )
    normalized_label_rows: list[str] = []
    normalized_to_label_rows: dict[str, list[str]] = {}
    for row in label_iter:
        norm = _normalize_path_text(row)
        if norm:
            normalized_label_rows.append(norm)
            normalized_to_label_rows.setdefault(norm, []).append(norm)

    image_counter = Counter(mapped_image_rows)
    label_counter = Counter(normalized_label_rows)
    all_keys = set(image_counter.keys()) | set(label_counter.keys())
    matched_count = sum(min(image_counter[k], label_counter[k]) for k in all_keys)

    unmatched_in_image = Counter(
        {k: image_counter[k] - label_counter[k] for k in all_keys if image_counter[k] > label_counter[k]}
    )
    unmatched_in_label = Counter(
        {k: label_counter[k] - image_counter[k] for k in all_keys if label_counter[k] > image_counter[k]}
    )
    unmatched_image_count = sum(unmatched_in_image.values())
    unmatched_label_count = sum(unmatched_in_label.values())

    matched_pairs: list[tuple[str, str]] = []
    for key in sorted(all_keys):
        image_items = mapped_to_image_rows.get(key, [])
        label_items = normalized_to_label_rows.get(key, [])
        pair_count = min(len(image_items), len(label_items))
        for i in range(pair_count):
            matched_pairs.append((image_items[i], label_items[i]))
    matched_pairs.sort(key=lambda x: x[0])

    unmatched_lines: list[str] = []
    for path_text, count in sorted(unmatched_in_image.items(), key=lambda x: x[0]):
        for _ in range(int(count)):
            unmatched_lines.append(path_text)
    for path_text, count in sorted(unmatched_in_label.items(), key=lambda x: x[0]):
        for _ in range(int(count)):
            unmatched_lines.append(path_text)

    out_unmatch_file.parent.mkdir(parents=True, exist_ok=True)
    with out_unmatch_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(unmatched_lines) + "\n")

    if out_match_pair_file is not None:
        out_match_pair_file.parent.mkdir(parents=True, exist_ok=True)
        with out_match_pair_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_file", "label_file"])
            writer.writerows(matched_pairs)

    sample_rows = unmatched_lines[:sample_limit]

    result = {
        "ok": True,
        "image_list_file": str(image_list_file),
        "label_list_file": str(label_list_file),
        "out_unmatch_file": str(out_unmatch_file),
        "out_match_pair_file": str(out_match_pair_file) if out_match_pair_file else None,
        "image_line_count": len(image_rows),
        "label_line_count": len(label_rows),
        "mapped_image_non_empty_count": len(mapped_image_rows),
        "label_non_empty_count": len(normalized_label_rows),
        "matched_count": matched_count,
        "unmatched_in_image_count": unmatched_image_count,
        "unmatched_in_label_count": unmatched_label_count,
        "keyword_mapping_count": len(keyword_mapping),
        "matched_pair_count": len(matched_pairs),
        "show_progress": show_progress,
        "sample_limit": sample_limit,
        "log_level": log_level,
    }
    logger.log(
        {
            **result,
            "keyword_mapping": keyword_mapping,
            "unmatched_sample_rows": sample_rows,
        },
        title="mapped path list match result",
    )
    emit(log_level, "INFO", f"[mapped-match] image={image_list_file}")
    emit(log_level, "INFO", f"[mapped-match] label={label_list_file}")
    emit(
        log_level,
        "INFO",
        f"[mapped-match] matched={matched_count}, unmatched_image={unmatched_image_count}, unmatched_label={unmatched_label_count}",
    )
    emit(log_level, "INFO", f"[mapped-match] unmatch_list={out_unmatch_file}")
    if out_match_pair_file is not None:
        emit(log_level, "INFO", f"[mapped-match] match_pair_csv={out_match_pair_file}")
    return result

