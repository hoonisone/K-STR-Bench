"""
경로 리스트(txt)의 확장자를 매핑(dict)으로 통일하고 실제 파일명도 동기화.

- list_file의 각 라인을 경로로 해석 (기본: dataset_dir 기준 상대경로)
- ext_mapping의 key 확장자 -> value 확장자로 변경
- 실제 파일 rename 수행 후 list_file 재저장 (out_list_file 지정 가능)
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _rename_case_only(src: Path, dst: Path) -> None:
    # Windows에서 대소문자만 바꾸는 rename 안정화를 위해 임시 이름 경유
    temp = src.with_name(src.name + f".__tmp_casefix__{uuid.uuid4().hex}")
    os.replace(src, temp)
    os.replace(temp, dst)


def _default_num_workers() -> int:
    return max(1, min(32, (os.cpu_count() or 4) * 2))


def _parse_parallelism(params: dict[str, Any]) -> tuple[int, int]:
    p_raw = params.get("num_process", params.get("num_proc", 1))
    num_process = int(p_raw)
    if num_process < 1:
        raise ValueError("num_process must be >= 1")

    t_raw = params.get(
        "num_thread",
        params.get("num_threads", params.get("num_worker", params.get("num_workers"))),
    )
    if t_raw is None:
        t_raw = _default_num_workers()
    num_thread = int(t_raw)
    if num_thread < 1:
        raise ValueError("num_thread (num_workers/num_worker) must be >= 1")
    return num_process, num_thread


def _split_tasks(
    tasks: list[tuple[int, str]],
    num_batches: int,
) -> list[list[tuple[int, str]]]:
    if not tasks:
        return []
    n = len(tasks)
    num_batches = max(1, min(num_batches, n))
    q, r = divmod(n, num_batches)
    batches: list[list[tuple[int, str]]] = []
    start = 0
    for i in range(num_batches):
        size = q + (1 if i < r else 0)
        end = start + size
        if start < end:
            batches.append(tasks[start:end])
        start = end
    return batches


def _normalize_ext_map(mapping_raw: dict[Any, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_key, raw_value in mapping_raw.items():
        src_ext = str(raw_key).strip().lower().lstrip(".")
        dst_ext = str(raw_value).strip().lower().lstrip(".")
        if not src_ext or not dst_ext:
            continue
        mapping[src_ext] = dst_ext
    if not mapping:
        raise ValueError("ext_mapping is empty after normalization")
    return mapping


def _load_mapping_csv(mapping_file: Path) -> dict[str, str]:
    if not mapping_file.is_file():
        raise FileNotFoundError(f"ext_mapping_file not found: {mapping_file}")
    with mapping_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(x or "").strip().lower() for x in (reader.fieldnames or [])]
        if not fieldnames:
            raise ValueError(f"invalid ext_mapping_file header: {mapping_file}")

        src_candidates = ("origin_dir", "from", "src", "old", "source", "origin")
        dst_candidates = ("new_dir", "to", "dst", "new", "target")
        src_col = next((c for c in src_candidates if c in fieldnames), None)
        dst_col = next((c for c in dst_candidates if c in fieldnames), None)
        if src_col is None or dst_col is None:
            raise ValueError(
                "ext_mapping_file must have source/target columns "
                f"(source candidates: {src_candidates}, target candidates: {dst_candidates})"
            )

        mapping_raw: dict[str, str] = {}
        for row in reader:
            src_val = str(row.get(src_col, "")).strip()
            dst_val = str(row.get(dst_col, "")).strip()
            if not src_val or not dst_val:
                continue
            mapping_raw[src_val] = dst_val
    return _normalize_ext_map(mapping_raw)


def _load_ext_mapping(params: dict[str, Any]) -> dict[str, str]:
    mapping_raw = params.get("ext_mapping", params.get("extension_mapping"))
    if mapping_raw is not None:
        if isinstance(mapping_raw, str):
            parsed = json.loads(mapping_raw)
            if not isinstance(parsed, dict):
                raise ValueError("ext_mapping JSON must be an object")
            return _normalize_ext_map(parsed)
        if not isinstance(mapping_raw, dict):
            raise ValueError("ext_mapping must be a mapping")
        return _normalize_ext_map(mapping_raw)

    mapping_file_raw = params.get("ext_mapping_file", params.get("mapping_file"))
    if mapping_file_raw:
        mapping_file = Path(str(mapping_file_raw)).expanduser().resolve()
        return _load_mapping_csv(mapping_file)

    raise ValueError("params.ext_mapping (or ext_mapping_file) is required")


def _normalize_rel_path_by_mapping(path_text: str, ext_map: dict[str, str]) -> str:
    normalized = path_text.replace("\\", "/")
    pp = PurePosixPath(normalized)
    if not pp.suffix:
        return normalized
    src_ext = pp.suffix.lstrip(".").lower()
    dst_ext = ext_map.get(src_ext)
    if not dst_ext:
        return normalized
    return str(pp.with_suffix(f".{dst_ext}")).replace("\\", "/")


def _resolve_abs_path(dataset_dir: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p.expanduser().resolve() if p.is_absolute() else (dataset_dir / p).resolve()


def _process_one_row(
    task: tuple[int, str],
    dataset_dir_str: str,
    ext_map: dict[str, str],
    skip_missing_files: bool,
    dry_run: bool,
) -> dict[str, Any]:
    idx, row = task
    trimmed = row.strip()
    if not trimmed:
        return {
            "idx": idx,
            "updated_row": "",
            "changed_line": False,
            "renamed": False,
            "missing": None,
            "conflict": None,
            "mapped": False,
        }

    normalized_trimmed = trimmed.replace("\\", "/")
    updated_rel = _normalize_rel_path_by_mapping(trimmed, ext_map=ext_map)
    mapped = updated_rel != normalized_trimmed

    # 문자열 기준으로 변경이 필요 없으면 파일시스템 접근 자체를 생략
    if not mapped:
        return {
            "idx": idx,
            "updated_row": normalized_trimmed,
            "changed_line": False,
            "renamed": False,
            "missing": None,
            "conflict": None,
            "mapped": False,
        }

    src_abs = _resolve_abs_path(Path(dataset_dir_str), trimmed)
    dst_abs = _resolve_abs_path(Path(dataset_dir_str), updated_rel)
    renamed = False
    missing: str | None = None
    conflict: str | None = None

    if src_abs != dst_abs:
        if not src_abs.exists():
            if skip_missing_files:
                missing = trimmed
            else:
                raise FileNotFoundError(f"source file not found: {src_abs}")
        else:
            if dst_abs.exists() and str(src_abs).lower() != str(dst_abs).lower():
                conflict = updated_rel
            else:
                if not dry_run:
                    dst_abs.parent.mkdir(parents=True, exist_ok=True)
                    if str(src_abs).lower() == str(dst_abs).lower():
                        _rename_case_only(src_abs, dst_abs)
                    else:
                        os.replace(src_abs, dst_abs)
                renamed = True

    return {
        "idx": idx,
        "updated_row": updated_rel,
        "changed_line": True,
        "renamed": renamed,
        "missing": missing,
        "conflict": conflict,
        "mapped": mapped,
    }


def _run_batch(
    batch_tasks: list[tuple[int, str]],
    dataset_dir_str: str,
    ext_map: dict[str, str],
    skip_missing_files: bool,
    dry_run: bool,
    num_thread: int,
    process_idx: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    def run_chunk(
        chunk_tasks: list[tuple[int, str]],
        thread_idx: int,
    ) -> list[dict[str, Any]]:
        iter_tasks: Any = (
            tqdm(
                chunk_tasks,
                desc="remap extensions [p0:t0]",
                unit="line",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_tasks
        )
        return [
            _process_one_row(
                task,
                dataset_dir_str,
                ext_map,
                skip_missing_files,
                dry_run,
            )
            for task in iter_tasks
        ]

    if num_thread <= 1:
        return run_chunk(batch_tasks, thread_idx=0)

    out: list[dict[str, Any]] = []
    thread_chunks = _split_tasks(batch_tasks, num_thread)
    with ThreadPoolExecutor(max_workers=num_thread) as ex:
        futures = [
            ex.submit(run_chunk, chunk, thread_idx)
            for thread_idx, chunk in enumerate(thread_chunks)
        ]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


class RemapPathListExtensionsStep:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        datadir: Any | None = None,
        list_file: Any | None = None,
        out_list_file: Any | None = None,
        ext_mapping: Any | None = None,
        extension_mapping: Any | None = None,
        ext_mapping_file: Any | None = None,
        mapping_file: Any | None = None,
        skip_missing_files: Any = True,
        dry_run: Any = False,
        show_progress: Any = True,
        num_process: Any = 1,
        num_thread: Any | None = None,
        num_workers: Any | None = None,
        num_worker: Any | None = None,
        log_level: Any | None = None,
        verbose: Any | None = None,
    ) -> None:
        self.params = {
            "dataset_dir": dataset_dir if dataset_dir is not None else datadir,
            "list_file": list_file,
            "out_list_file": out_list_file,
            "ext_mapping": ext_mapping if ext_mapping is not None else extension_mapping,
            "ext_mapping_file": (
                ext_mapping_file if ext_mapping_file is not None else mapping_file
            ),
            "skip_missing_files": skip_missing_files,
            "dry_run": dry_run,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": (
                num_thread
                if num_thread is not None
                else (num_worker if num_worker is not None else num_workers)
            ),
            "log_level": log_level,
            "verbose": verbose,
        }

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    if not dataset_dir_raw:
        raise ValueError("params.dataset_dir (or datadir) is required")
    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    list_file_raw = params.get("list_file")
    if not list_file_raw:
        raise ValueError("params.list_file is required")
    list_file = Path(str(list_file_raw)).expanduser().resolve()
    if not list_file.is_file():
        raise FileNotFoundError(f"list_file not found: {list_file}")

    out_list_file_raw = params.get("out_list_file")
    out_list_file = (
        Path(str(out_list_file_raw)).expanduser().resolve()
        if out_list_file_raw
        else list_file
    )

    ext_map = _load_ext_mapping(params)
    skip_missing_files = coerce_bool(params.get("skip_missing_files", True), default=True)
    dry_run = coerce_bool(params.get("dry_run", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    num_process, num_thread = _parse_parallelism(params)

    if params.get("log_level") is not None:
        log_level = normalize_log_level(params.get("log_level"), default="INFO")
    elif params.get("verbose") is not None:
        log_level = "INFO" if coerce_bool(params.get("verbose"), default=True) else "WARNING"
    else:
        log_level = "INFO"

    logger = StepLogger("remap_path_list_extensions_entry")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "list_file": str(list_file),
            "out_list_file": str(out_list_file),
            "dry_run": dry_run,
            "show_progress": show_progress,
            "skip_missing_files": skip_missing_files,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
            "mapping_count": len(ext_map),
            "mapping_preview": list(ext_map.items())[:10],
        },
        title="remap discovery",
    )
    emit(log_level, "INFO", f"[remap-ext] mapping_count={len(ext_map)}")
    emit(
        log_level,
        "INFO",
        f"[remap-ext] workers process={num_process}, thread={num_thread}, dry_run={dry_run}",
    )

    with list_file.open("r", encoding="utf-8") as f:
        original_rows = [line.rstrip("\n") for line in f]

    tasks = [(idx, row) for idx, row in enumerate(original_rows)]
    dataset_dir_str = str(dataset_dir)

    if num_process > 1:
        batches = _split_tasks(tasks, num_process)
        with ProcessPoolExecutor(max_workers=num_process) as ex:
            futures = [
                ex.submit(
                    _run_batch,
                    batch,
                    dataset_dir_str,
                    ext_map,
                    skip_missing_files,
                    dry_run,
                    num_thread,
                    process_idx,
                    show_progress,
                )
                for process_idx, batch in enumerate(batches)
            ]
            results: list[dict[str, Any]] = []
            for fut in as_completed(futures):
                results.extend(fut.result())
    else:
        results = _run_batch(
            batch_tasks=tasks,
            dataset_dir_str=dataset_dir_str,
            ext_map=ext_map,
            skip_missing_files=skip_missing_files,
            dry_run=dry_run,
            num_thread=num_thread,
            process_idx=0,
            show_progress=show_progress,
        )

    results.sort(key=lambda r: int(r["idx"]))
    updated_rows: list[str] = [""] * len(original_rows)
    renamed_count = 0
    changed_line_count = 0
    mapped_line_count = 0
    missing_files: list[str] = []
    conflict_files: list[str] = []
    sample_renames: list[tuple[str, str]] = []

    for r in results:
        idx = int(r["idx"])
        updated_rows[idx] = str(r["updated_row"])
        if bool(r["changed_line"]):
            changed_line_count += 1
        if bool(r["mapped"]):
            mapped_line_count += 1
        if bool(r["renamed"]):
            renamed_count += 1
            if len(sample_renames) < 10:
                sample_renames.append((original_rows[idx].strip(), updated_rows[idx]))
        if r["missing"]:
            missing_files.append(str(r["missing"]))
        if r["conflict"]:
            conflict_files.append(str(r["conflict"]))

    if not dry_run:
        out_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for row in updated_rows:
                f.write(row + "\n")

    summary = {
        "ok": True,
        "dataset_dir": str(dataset_dir),
        "list_file": str(list_file),
        "out_list_file": str(out_list_file),
        "dry_run": dry_run,
        "show_progress": show_progress,
        "skip_missing_files": skip_missing_files,
        "num_process": num_process,
        "num_thread": num_thread,
        "mapping_count": len(ext_map),
        "line_count": len(original_rows),
        "mapped_line_count": mapped_line_count,
        "changed_line_count": changed_line_count,
        "renamed_count": renamed_count,
        "missing_file_count": len(missing_files),
        "conflict_count": len(conflict_files),
        "sample_renames": sample_renames,
    }
    logger.log(summary, title="remap summary")
    emit(
        log_level,
        "INFO",
        (
            f"[remap-ext] done lines={len(original_rows)}, mapped={mapped_line_count}, "
            f"renamed={renamed_count}, missing={len(missing_files)}, conflict={len(conflict_files)}"
        ),
    )
    return summary

