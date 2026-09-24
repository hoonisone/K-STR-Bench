"""
매칭된 image/label pair CSV를 기반으로 전역 ID 파일명 매핑 CSV를 생성.

- 입력 CSV 컬럼: image_file, label_file (기본값)
- 각 pair에 대해 동일한 전역 ID를 부여 (기본: 1, 2, 3, ...)
- `id_prefix`가 있으면 파일명이 `{prefix}_{id}`가 됨 (예: aihub_1.jpg)
- 파일명만 ID로 치환하고 확장자는 유지
- 출력 CSV는 행 단위(flat)로 저장: origin_path,new_path
"""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_path_text(path_text: str) -> str:
    return path_text.strip().replace("\\", "/")


def _as_posix(path_obj: PurePosixPath | Path) -> str:
    return str(path_obj).replace("\\", "/")


def _build_new_path(path_text: str, id_text: str) -> str:
    norm = _normalize_path_text(path_text)
    if not norm:
        return ""
    pp = PurePosixPath(norm)
    new_name = f"{id_text}{pp.suffix}"
    return _as_posix(pp.with_name(new_name))


class MakeGlobalIdRenameMapFromMatchPairStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    pair_csv_file_raw = params.get("match_pair_csv_file")
    out_csv_file_raw = params.get("out_mapping_csv_file")
    image_col = str(params.get("image_col", "image_file"))
    label_col = str(params.get("label_col", "label_file"))
    origin_col = str(params.get("origin_col", "origin_path"))
    new_col = str(params.get("new_col", "new_path"))
    start_id = int(params.get("start_id", 1))
    id_padding = int(params.get("id_padding", 0))
    id_prefix = str(params.get("id_prefix", "") or "").strip().rstrip("_")
    normalize_slashes = coerce_bool(params.get("normalize_slashes", True), default=True)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not pair_csv_file_raw:
        raise ValueError("match_pair_csv_file is required")
    if not out_csv_file_raw:
        raise ValueError("out_mapping_csv_file is required")
    if start_id < 0:
        raise ValueError("start_id must be >= 0")
    if id_padding < 0:
        raise ValueError("id_padding must be >= 0")

    pair_csv_file = Path(str(pair_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not pair_csv_file.is_file():
        raise FileNotFoundError(f"match_pair_csv_file not found: {pair_csv_file}")

    logger = StepLogger("make_global_id_rename_map_from_match_pair_entry")
    logger.log(
        {
            "match_pair_csv_file": str(pair_csv_file),
            "out_mapping_csv_file": str(out_csv_file),
            "image_col": image_col,
            "label_col": label_col,
            "origin_col": origin_col,
            "new_col": new_col,
            "start_id": start_id,
            "id_padding": id_padding,
            "id_prefix": id_prefix,
            "normalize_slashes": normalize_slashes,
            "log_level": log_level,
        },
        title="global id rename map discovery",
    )

    with pair_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if image_col not in fieldnames or label_col not in fieldnames:
            raise ValueError(
                f"CSV must include '{image_col}' and '{label_col}': {pair_csv_file}"
            )
        pair_rows = [dict(row) for row in reader]

    mapping_rows: list[dict[str, str]] = []
    duplicate_new_path_count = 0
    seen_new_paths: set[str] = set()

    for idx, row in enumerate(pair_rows):
        pair_id = start_id + idx
        id_text = str(pair_id).zfill(id_padding) if id_padding > 0 else str(pair_id)
        if id_prefix:
            id_text = f"{id_prefix}_{id_text}"
        image_path = str(row.get(image_col, "") or "")
        label_path = str(row.get(label_col, "") or "")

        image_origin = _normalize_path_text(image_path) if normalize_slashes else image_path.strip()
        label_origin = _normalize_path_text(label_path) if normalize_slashes else label_path.strip()
        image_new = _build_new_path(image_origin, id_text)
        label_new = _build_new_path(label_origin, id_text)

        if image_origin:
            mapping_rows.append({origin_col: image_origin, new_col: image_new})
            if image_new in seen_new_paths:
                duplicate_new_path_count += 1
            seen_new_paths.add(image_new)
        if label_origin:
            mapping_rows.append({origin_col: label_origin, new_col: label_new})
            if label_new in seen_new_paths:
                duplicate_new_path_count += 1
            seen_new_paths.add(label_new)

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[origin_col, new_col])
        writer.writeheader()
        writer.writerows(mapping_rows)

    result = {
        "ok": True,
        "match_pair_csv_file": str(pair_csv_file),
        "out_mapping_csv_file": str(out_csv_file),
        "pair_count": len(pair_rows),
        "mapping_row_count": len(mapping_rows),
        "start_id": start_id,
        "last_id": (start_id + len(pair_rows) - 1) if pair_rows else None,
        "duplicate_new_path_count": duplicate_new_path_count,
        "image_col": image_col,
        "label_col": label_col,
        "origin_col": origin_col,
        "new_col": new_col,
        "id_padding": id_padding,
        "id_prefix": id_prefix,
        "normalize_slashes": normalize_slashes,
        "log_level": log_level,
    }
    logger.log(result, title="global id rename map summary")

    emit(log_level, "INFO", f"[global-id-map] pair_csv={pair_csv_file}")
    emit(log_level, "INFO", f"[global-id-map] out_csv={out_csv_file}")
    emit(
        log_level,
        "INFO",
        (
            f"[global-id-map] pair_count={len(pair_rows)}, "
            f"mapping_rows={len(mapping_rows)}, duplicate_new_path_count={duplicate_new_path_count}"
        ),
    )
    return result

