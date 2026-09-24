from __future__ import annotations

import csv
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _rename_case_only(src: Path, dst: Path) -> None:
    temp = src.with_name(src.name + f".__tmp_casefix__{uuid.uuid4().hex}")
    os.replace(src, temp)
    os.replace(temp, dst)


def _resolve_abs(base_dir: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p.expanduser().resolve() if p.is_absolute() else (base_dir / p).resolve()


def _parse_parallelism(params: dict[str, Any]) -> tuple[int, int]:
    p_raw = params.get("num_process", params.get("num_proc", 1))
    t_raw = params.get(
        "num_thread",
        params.get("num_threads", params.get("num_worker", params.get("num_workers", 1))),
    )
    num_process = int(p_raw)
    num_thread = int(t_raw)
    if num_process < 1:
        raise ValueError("num_process must be >= 1")
    if num_thread < 1:
        raise ValueError("num_thread must be >= 1")
    return num_process, num_thread


def _split_tasks(tasks: list[dict[str, str]], num_batches: int) -> list[list[dict[str, str]]]:
    if not tasks:
        return []
    n = len(tasks)
    num_batches = max(1, min(num_batches, n))
    q, r = divmod(n, num_batches)
    out: list[list[dict[str, str]]] = []
    start = 0
    for i in range(num_batches):
        size = q + (1 if i < r else 0)
        end = start + size
        if start < end:
            out.append(tasks[start:end])
        start = end
    return out


def _process_one_row(
    row: dict[str, str],
    dataset_dir_str: str,
    out_dataset_dir_str: str,
    origin_col: str,
    new_col: str,
    dry_run: bool,
    skip_missing_files: bool,
) -> dict[str, Any]:
    origin_raw = str(row.get(origin_col, "") or "").strip().replace("\\", "/")
    new_raw = str(row.get(new_col, "") or "").strip().replace("\\", "/")
    if not origin_raw or not new_raw:
        return {
            "invalid": True,
            "renamed": False,
            "unchanged": False,
            "missing": None,
            "conflict": None,
            "origin": origin_raw,
            "new": new_raw,
        }

    src_abs = _resolve_abs(Path(dataset_dir_str), origin_raw)
    dst_abs = _resolve_abs(Path(out_dataset_dir_str), new_raw)

    if src_abs == dst_abs:
        return {
            "invalid": False,
            "renamed": False,
            "unchanged": True,
            "missing": None,
            "conflict": None,
            "origin": origin_raw,
            "new": new_raw,
        }

    if not src_abs.exists():
        # Already applied on a previous run: source was moved, dest remains.
        if dst_abs.exists():
            return {
                "invalid": False,
                "renamed": False,
                "unchanged": True,
                "already_applied": True,
                "missing": None,
                "conflict": None,
                "origin": origin_raw,
                "new": new_raw,
            }
        if skip_missing_files:
            return {
                "invalid": False,
                "renamed": False,
                "unchanged": False,
                "missing": origin_raw,
                "conflict": None,
                "origin": origin_raw,
                "new": new_raw,
            }
        raise FileNotFoundError(f"source path not found: {src_abs}")

    if dst_abs.exists() and str(src_abs).lower() != str(dst_abs).lower():
        return {
            "invalid": False,
            "renamed": False,
            "unchanged": False,
            "missing": None,
            "conflict": new_raw,
            "origin": origin_raw,
            "new": new_raw,
        }

    if not dry_run:
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        if str(src_abs).lower() == str(dst_abs).lower():
            _rename_case_only(src_abs, dst_abs)
        else:
            # 파일/폴더 모두 허용
            shutil.move(str(src_abs), str(dst_abs))
    return {
        "invalid": False,
        "renamed": True,
        "unchanged": False,
        "missing": None,
        "conflict": None,
        "origin": origin_raw,
        "new": new_raw,
    }


def _run_batch(
    batch_rows: list[dict[str, str]],
    dataset_dir_str: str,
    out_dataset_dir_str: str,
    origin_col: str,
    new_col: str,
    dry_run: bool,
    skip_missing_files: bool,
    num_thread: int,
    process_idx: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    def run_chunk(chunk_rows: list[dict[str, str]], thread_idx: int) -> list[dict[str, Any]]:
        iter_rows: Any = (
            tqdm(
                chunk_rows,
                desc="apply path mapping [p0:t0]",
                unit="row",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_rows
        )
        return [
            _process_one_row(
                row=row,
                dataset_dir_str=dataset_dir_str,
                out_dataset_dir_str=out_dataset_dir_str,
                origin_col=origin_col,
                new_col=new_col,
                dry_run=dry_run,
                skip_missing_files=skip_missing_files,
            )
            for row in iter_rows
        ]

    if num_thread <= 1:
        return run_chunk(batch_rows, thread_idx=0)

    chunks = _split_tasks(batch_rows, num_thread)
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=num_thread) as ex:
        futures = [ex.submit(run_chunk, chunk, idx) for idx, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


class ApplyPathMappingCsvStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    out_dataset_dir_raw = params.get("out_dataset_dir")
    mapping_csv_file_raw = (
        params.get("mapping_csv_file")
        or params.get("file_mapping_csv_file")
        or params.get("out_mapping_csv_file")
    )
    origin_col = str(params.get("origin_col", "origin_path"))
    new_col = str(params.get("new_col", "new_path"))
    out_changed_list_file_raw = params.get("out_changed_list_file")
    dry_run = coerce_bool(params.get("dry_run", False), default=False)
    skip_missing_files = coerce_bool(params.get("skip_missing_files", True), default=True)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    num_process, num_thread = _parse_parallelism(params)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not dataset_dir_raw:
        raise ValueError("dataset_dir (or datadir) is required")
    if not mapping_csv_file_raw:
        raise ValueError("mapping_csv_file (or out_mapping_csv_file) is required")

    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    out_dataset_dir = (
        Path(str(out_dataset_dir_raw)).expanduser().resolve()
        if out_dataset_dir_raw
        else dataset_dir
    )
    mapping_csv_file = Path(str(mapping_csv_file_raw)).expanduser().resolve()
    out_changed_list_file = (
        Path(str(out_changed_list_file_raw)).expanduser().resolve()
        if out_changed_list_file_raw
        else None
    )

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    if not mapping_csv_file.is_file():
        raise FileNotFoundError(f"mapping_csv_file not found: {mapping_csv_file}")
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    logger = StepLogger("apply_path_mapping_csv_entry")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "mapping_csv_file": str(mapping_csv_file),
            "origin_col": origin_col,
            "new_col": new_col,
            "out_changed_list_file": (
                str(out_changed_list_file) if out_changed_list_file else None
            ),
            "dry_run": dry_run,
            "skip_missing_files": skip_missing_files,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
        },
        title="apply path mapping csv discovery",
    )

    with mapping_csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if origin_col not in fieldnames or new_col not in fieldnames:
            raise ValueError(
                f"CSV must include '{origin_col}' and '{new_col}': {mapping_csv_file}"
            )
        rows = [dict(r) for r in reader]

    if num_process > 1:
        batches = _split_tasks(rows, num_process)
        with ProcessPoolExecutor(max_workers=num_process) as ex:
            futures = [
                ex.submit(
                    _run_batch,
                    batch,
                    str(dataset_dir),
                    str(out_dataset_dir),
                    origin_col,
                    new_col,
                    dry_run,
                    skip_missing_files,
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
            batch_rows=rows,
            dataset_dir_str=str(dataset_dir),
            out_dataset_dir_str=str(out_dataset_dir),
            origin_col=origin_col,
            new_col=new_col,
            dry_run=dry_run,
            skip_missing_files=skip_missing_files,
            num_thread=num_thread,
            process_idx=0,
            show_progress=show_progress,
        )

    renamed_count = 0
    unchanged_count = 0
    missing_count = 0
    conflict_count = 0
    invalid_row_count = 0
    sample_renamed: list[tuple[str, str]] = []
    changed_path_list: list[str] = []
    sample_missing: list[str] = []
    sample_conflict: list[str] = []
    for r in results:
        origin_raw = str(r.get("origin", "") or "")
        new_raw = str(r.get("new", "") or "")
        if bool(r.get("invalid")):
            invalid_row_count += 1
            continue
        if bool(r.get("unchanged")):
            unchanged_count += 1
            if r.get("already_applied") and new_raw:
                changed_path_list.append(new_raw)
            continue
        if r.get("missing"):
            missing_count += 1
            if len(sample_missing) < 20:
                sample_missing.append(str(r["missing"]))
            continue
        if r.get("conflict"):
            conflict_count += 1
            if len(sample_conflict) < 20:
                sample_conflict.append(str(r["conflict"]))
            continue
        if bool(r.get("renamed")):
            renamed_count += 1
            if len(sample_renamed) < 20:
                sample_renamed.append((origin_raw, new_raw))
            changed_path_list.append(new_raw)

    if out_changed_list_file is not None:
        out_changed_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_changed_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for path_text in changed_path_list:
                f.write(path_text + "\n")

    summary = {
        "ok": True,
        "dataset_dir": str(dataset_dir),
        "out_dataset_dir": str(out_dataset_dir),
        "mapping_csv_file": str(mapping_csv_file),
        "out_changed_list_file": (
            str(out_changed_list_file) if out_changed_list_file else None
        ),
        "origin_col": origin_col,
        "new_col": new_col,
        "dry_run": dry_run,
        "skip_missing_files": skip_missing_files,
        "show_progress": show_progress,
        "num_process": num_process,
        "num_thread": num_thread,
        "row_count": len(rows),
        "renamed_count": renamed_count,
        "unchanged_count": unchanged_count,
        "missing_count": missing_count,
        "conflict_count": conflict_count,
        "invalid_row_count": invalid_row_count,
        "sample_renamed": sample_renamed,
        "saved_changed_list_count": len(changed_path_list)
        if out_changed_list_file is not None
        else None,
        "sample_missing": sample_missing,
        "sample_conflict": sample_conflict,
        "log_level": log_level,
    }
    logger.log(summary, title="apply path mapping csv summary")
    emit(
        log_level,
        "INFO",
        (
            f"[apply-path-map] rows={len(rows)}, renamed={renamed_count}, unchanged={unchanged_count}, "
            f"missing={missing_count}, conflict={conflict_count}, invalid={invalid_row_count}"
        ),
    )
    if out_changed_list_file is not None:
        emit(log_level, "INFO", f"[apply-path-map] out_changed_list_file={out_changed_list_file}")
    return summary

