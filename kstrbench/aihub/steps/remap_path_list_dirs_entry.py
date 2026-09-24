"""
경로 리스트(txt)의 디렉터리 prefix를 매핑(csv)으로 치환하고 실제 파일 경로도 동기화.

- list_file 각 라인(기본: dataset_dir 기준 상대경로)에 대해
  origin_dir -> new_dir 매핑(prefix 기준) 적용
- 실제 파일을 새 경로로 이동(rename)하고 리스트를 갱신 저장
"""

from __future__ import annotations

import csv
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm import tqdm

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _rename_case_only(src: Path, dst: Path) -> None:
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


def _normalize_rel_path(path_text: str) -> str:
    return str(PurePosixPath(path_text.strip().replace("\\", "/")))


def _load_path_mapping(
    mapping_file: Path,
    origin_col_name: str | None = None,
    new_col_name: str | None = None,
) -> list[tuple[str, str]]:
    if not mapping_file.is_file():
        raise FileNotFoundError(f"dir_mapping_file not found: {mapping_file}")

    rules: list[tuple[str, str]] = []
    with mapping_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"invalid dir_mapping header: {mapping_file}")
        fields = {str(name).strip().lower(): name for name in reader.fieldnames}
        if origin_col_name:
            origin_col = fields.get(origin_col_name.strip().lower())
        else:
            origin_col = fields.get("origin_path") or fields.get("origin_dir")
        if new_col_name:
            new_col = fields.get(new_col_name.strip().lower())
        else:
            new_col = fields.get("new_path") or fields.get("new_dir")
        if not origin_col or not new_col:
            raise ValueError(
                "dir_mapping_file must contain columns: "
                "(origin_path, new_path) or (origin_dir, new_dir)"
            )

        for row in reader:
            origin_raw = str(row.get(origin_col, "")).strip()
            new_raw = str(row.get(new_col, "")).strip()
            if not origin_raw or not new_raw:
                continue
            origin = _normalize_rel_path(origin_raw)
            new = _normalize_rel_path(new_raw)
            rules.append((origin, new))

    if not rules:
        raise ValueError(f"no valid mapping rows: {mapping_file}")

    # 더 긴 prefix 먼저 매칭
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules


def _apply_dir_mapping(path_text: str, rules: list[tuple[str, str]]) -> tuple[str, bool]:
    rel = _normalize_rel_path(path_text)
    for origin, new in rules:
        if rel == origin:
            return new, True
        if rel.startswith(origin + "/"):
            suffix = rel[len(origin) + 1 :]
            return f"{new}/{suffix}", True
    return rel, False


def _resolve_abs_path(dataset_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p.expanduser().resolve() if p.is_absolute() else (dataset_dir / p).resolve()


def _process_one_row(
    task: tuple[int, str],
    dataset_dir_str: str,
    rules: list[tuple[str, str]],
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

    updated_rel, mapped = _apply_dir_mapping(trimmed, rules)
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
                        shutil.move(str(src_abs), str(dst_abs))
                renamed = True

    return {
        "idx": idx,
        "updated_row": updated_rel,
        "changed_line": updated_rel != _normalize_rel_path(trimmed),
        "renamed": renamed,
        "missing": missing,
        "conflict": conflict,
        "mapped": mapped,
    }


def _run_batch(
    batch_tasks: list[tuple[int, str]],
    dataset_dir_str: str,
    rules: list[tuple[str, str]],
    skip_missing_files: bool,
    dry_run: bool,
    num_thread: int,
    process_idx: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    def run_chunk(chunk_tasks: list[tuple[int, str]], thread_idx: int) -> list[dict[str, Any]]:
        iter_tasks: Any = (
            tqdm(
                chunk_tasks,
                desc="remap dirs [p0:t0]",
                unit="line",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_tasks
        )
        return [
            _process_one_row(
                task=task,
                dataset_dir_str=dataset_dir_str,
                rules=rules,
                skip_missing_files=skip_missing_files,
                dry_run=dry_run,
            )
            for task in iter_tasks
        ]

    if num_thread <= 1:
        return run_chunk(batch_tasks, thread_idx=0)

    out: list[dict[str, Any]] = []
    chunks = _split_tasks(batch_tasks, num_thread)
    with ThreadPoolExecutor(max_workers=num_thread) as ex:
        futures = [ex.submit(run_chunk, chunk, i) for i, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


class RemapPathListDirsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _move_paths_by_mapping(
    src_dataset_dir: Path,
    out_dataset_dir: Path,
    rules: list[tuple[str, str]],
    dry_run: bool,
    path_kind: str,
) -> dict[str, Any]:
    moved_count = 0
    missing_source_count = 0
    conflict_count = 0
    moved_samples: list[tuple[str, str]] = []
    moved_path_list: list[str] = []
    missing_samples: list[str] = []
    conflict_samples: list[str] = []

    # 더 긴 경로부터 이동 (중첩된 origin 충돌 완화)
    ordered_rules = sorted(rules, key=lambda x: len(Path(x[0]).parts), reverse=True)

    for origin_rel, new_rel in ordered_rules:
        src = (src_dataset_dir / Path(origin_rel)).resolve()
        dst = (out_dataset_dir / Path(new_rel)).resolve()

        if not src.exists():
            missing_source_count += 1
            if len(missing_samples) < 20:
                missing_samples.append(origin_rel)
            continue
        if path_kind == "dir" and not src.is_dir():
            conflict_count += 1
            if len(conflict_samples) < 20:
                conflict_samples.append(f"source is not directory: {origin_rel}")
            continue
        if path_kind == "file" and not src.is_file():
            conflict_count += 1
            if len(conflict_samples) < 20:
                conflict_samples.append(f"source is not file: {origin_rel}")
            continue

        dst_parent = dst.parent
        if not dry_run:
            dst_parent.mkdir(parents=True, exist_ok=True)

        # case-only 변경은 임시 이름 경유
        if str(src).lower() == str(dst).lower():
            if not dry_run:
                _rename_case_only(src, dst)
            moved_count += 1
            moved_path_list.append(new_rel)
            if len(moved_samples) < 20:
                moved_samples.append((origin_rel, new_rel))
            continue

        if dst.exists():
            conflict_count += 1
            if len(conflict_samples) < 20:
                conflict_samples.append(f"target exists: {new_rel}")
            continue

        if not dry_run:
            shutil.move(str(src), str(dst))
        moved_count += 1
        moved_path_list.append(new_rel)
        if len(moved_samples) < 20:
            moved_samples.append((origin_rel, new_rel))

    return {
        "moved_count": moved_count,
        "moved_path_list": moved_path_list,
        "missing_source_count": missing_source_count,
        "conflict_count": conflict_count,
        "moved_samples": moved_samples,
        "missing_samples": missing_samples,
        "conflict_samples": conflict_samples,
    }


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    out_dataset_dir_raw = params.get("out_dataset_dir", params.get("target_dataset_dir"))
    source_path_list_file_raw = params.get(
        "source_path_list_file", params.get("source_dir_list_file")
    )
    out_path_list_file_raw = params.get(
        "out_path_list_file", params.get("out_dir_list_file")
    )
    out_changed_list_file_raw = params.get("out_changed_list_file")
    list_file_raw = params.get("list_file")
    out_list_file_raw = params.get("out_list_file")
    dir_mapping_file_raw = params.get("dir_mapping_file", params.get("mapping_file"))
    origin_col = params.get("origin_col")
    new_col = params.get("new_col")
    if not dataset_dir_raw:
        raise ValueError("dataset_dir (or datadir) is required")
    if not dir_mapping_file_raw:
        raise ValueError("dir_mapping_file (or mapping_file) is required")
    path_type_raw = params.get("path_type", params.get("path_kind"))
    if path_type_raw is None and "move_directories" in params:
        # backward compatibility: move_directories=true였던 기존 설정은 dir로 취급
        path_type_raw = "dir"
    path_type = str(path_type_raw if path_type_raw is not None else "dir").strip().lower()
    if path_type not in {"dir", "file"}:
        raise ValueError("path_type must be one of: dir, file")
    use_mapping_list_mode = source_path_list_file_raw is not None
    use_path_list_mode = list_file_raw is not None

    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    out_dataset_dir = (
        Path(str(out_dataset_dir_raw)).expanduser().resolve()
        if out_dataset_dir_raw
        else dataset_dir
    )
    source_path_list_file = (
        Path(str(source_path_list_file_raw)).expanduser().resolve()
        if source_path_list_file_raw is not None
        else None
    )
    out_path_list_file = (
        Path(str(out_path_list_file_raw)).expanduser().resolve()
        if out_path_list_file_raw is not None
        else None
    )
    out_changed_list_file = (
        Path(str(out_changed_list_file_raw)).expanduser().resolve()
        if out_changed_list_file_raw is not None
        else None
    )
    list_file = Path(str(list_file_raw)).expanduser().resolve() if list_file_raw else None
    out_list_file = None
    if list_file is not None:
        out_list_file = (
            Path(str(out_list_file_raw)).expanduser().resolve()
            if out_list_file_raw
            else list_file
        )
    dir_mapping_file = Path(str(dir_mapping_file_raw)).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    if source_path_list_file is not None and not source_path_list_file.is_file():
        raise FileNotFoundError(f"source_path_list_file not found: {source_path_list_file}")
    if list_file is not None and not list_file.is_file():
        raise FileNotFoundError(f"list_file not found: {list_file}")

    rules = _load_path_mapping(
        dir_mapping_file,
        origin_col_name=str(origin_col) if origin_col is not None else None,
        new_col_name=str(new_col) if new_col is not None else None,
    )
    skip_missing_files = coerce_bool(params.get("skip_missing_files", True), default=True)
    dry_run = coerce_bool(params.get("dry_run", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    num_process, num_thread = _parse_parallelism(params)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    logger = StepLogger("remap_path_list_dirs_entry")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "list_file": str(list_file) if list_file is not None else None,
            "out_list_file": str(out_list_file) if out_list_file is not None else None,
            "source_path_list_file": (
                str(source_path_list_file) if source_path_list_file else None
            ),
            "out_path_list_file": str(out_path_list_file) if out_path_list_file else None,
            "out_changed_list_file": (
                str(out_changed_list_file) if out_changed_list_file else None
            ),
            "dir_mapping_file": str(dir_mapping_file),
            "origin_col": str(origin_col) if origin_col is not None else None,
            "new_col": str(new_col) if new_col is not None else None,
            "path_type": path_type,
            "use_mapping_list_mode": use_mapping_list_mode,
            "use_path_list_mode": use_path_list_mode,
            "mapping_count": len(rules),
            "mapping_preview": rules[:10],
            "skip_missing_files": skip_missing_files,
            "dry_run": dry_run,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
        },
        title="remap dirs discovery",
    )

    if use_mapping_list_mode:
        if source_path_list_file is None:
            raise ValueError(
                "source_path_list_file is required in mapping-list mode"
            )

        origin_to_new: dict[str, str] = {}
        duplicate_origin_conflicts: list[str] = []
        for origin_rel, new_rel in rules:
            prev = origin_to_new.get(origin_rel)
            if prev is not None and prev != new_rel:
                duplicate_origin_conflicts.append(origin_rel)
            origin_to_new[origin_rel] = new_rel
        if duplicate_origin_conflicts:
            raise RuntimeError(
                "dir_mapping_file has duplicate origin_dir with different new_dir: "
                f"{duplicate_origin_conflicts[:10]}"
            )

        dir_list_rows: list[str] = []
        with source_path_list_file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip().replace("\\", "/")
                if not line:
                    continue
                dir_list_rows.append(line)

        not_mapped_rows = [row for row in dir_list_rows if row not in origin_to_new]
        if not_mapped_rows:
            raise RuntimeError(
                "source_path_list_file contains paths not found in dir_mapping origin_path: "
                f"count={len(not_mapped_rows)}, sample={not_mapped_rows[:20]}"
            )

        selected_rules: list[tuple[str, str]] = []
        seen_origin: set[str] = set()
        for origin_rel in dir_list_rows:
            if origin_rel in seen_origin:
                continue
            seen_origin.add(origin_rel)
            selected_rules.append((origin_rel, origin_to_new[origin_rel]))

        move_summary = _move_paths_by_mapping(
            src_dataset_dir=dataset_dir,
            out_dataset_dir=out_dataset_dir,
            rules=selected_rules,
            dry_run=dry_run,
            path_kind=path_type,
        )
        saved_dir_list_count: int | None = None
        mapped_dir_list_count: int | None = None
        if out_path_list_file is not None:
            out_path_list_file.parent.mkdir(parents=True, exist_ok=True)
            mapped_dirs = [origin_to_new[row] for row in dir_list_rows]
            mapped_dir_list_count = len(mapped_dirs)
            with out_path_list_file.open("w", encoding="utf-8", newline="\n") as f:
                for rel in mapped_dirs:
                    f.write(rel + "\n")
            saved_dir_list_count = len(mapped_dirs)
        saved_changed_list_count: int | None = None
        if out_changed_list_file is not None:
            out_changed_list_file.parent.mkdir(parents=True, exist_ok=True)
            with out_changed_list_file.open("w", encoding="utf-8", newline="\n") as f:
                for rel in move_summary["moved_path_list"]:
                    f.write(rel + "\n")
            saved_changed_list_count = len(move_summary["moved_path_list"])
        summary = {
            "ok": True,
            "mode": "mapping_list",
            "dataset_dir": str(dataset_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "source_path_list_file": (
                str(source_path_list_file) if source_path_list_file else None
            ),
            "out_path_list_file": str(out_path_list_file) if out_path_list_file else None,
            "out_changed_list_file": (
                str(out_changed_list_file) if out_changed_list_file else None
            ),
            "saved_dir_list_count": saved_dir_list_count,
            "saved_changed_list_count": saved_changed_list_count,
            "mapped_dir_list_count": mapped_dir_list_count,
            "source_dir_list_count": len(dir_list_rows),
            "selected_mapping_count": len(selected_rules),
            "dir_mapping_file": str(dir_mapping_file),
            "mapping_count": len(rules),
            "path_type": path_type,
            "dry_run": dry_run,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": num_thread,
            "moved_count": move_summary["moved_count"],
            "missing_source_count": move_summary["missing_source_count"],
            "conflict_count": move_summary["conflict_count"],
            "moved_samples": move_summary["moved_samples"],
            "missing_samples": move_summary["missing_samples"],
            "conflict_samples": move_summary["conflict_samples"],
            "log_level": log_level,
        }
        logger.log(summary, title="remap dirs mapping-list summary")
        emit(
            log_level,
            "INFO",
            (
                f"[remap-dirs] mode=mapping_list path_type={path_type} moved={move_summary['moved_count']}, "
                f"missing={move_summary['missing_source_count']}, conflict={move_summary['conflict_count']}"
            ),
        )
        if out_path_list_file is not None:
            emit(log_level, "INFO", f"[remap-dirs] out_path_list_file={out_path_list_file}")
        if out_changed_list_file is not None:
            emit(
                log_level,
                "INFO",
                f"[remap-dirs] out_changed_list_file={out_changed_list_file}",
            )
        return summary

    if not use_path_list_mode:
        move_summary = _move_paths_by_mapping(
            src_dataset_dir=dataset_dir,
            out_dataset_dir=out_dataset_dir,
            rules=rules,
            dry_run=dry_run,
            path_kind=path_type,
        )
        saved_changed_list_count: int | None = None
        if out_changed_list_file is not None:
            out_changed_list_file.parent.mkdir(parents=True, exist_ok=True)
            with out_changed_list_file.open("w", encoding="utf-8", newline="\n") as f:
                for rel in move_summary["moved_path_list"]:
                    f.write(rel + "\n")
            saved_changed_list_count = len(move_summary["moved_path_list"])
        summary = {
            "ok": True,
            "mode": "mapping_csv",
            "dataset_dir": str(dataset_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "source_path_list_file": None,
            "out_path_list_file": None,
            "out_changed_list_file": (
                str(out_changed_list_file) if out_changed_list_file else None
            ),
            "saved_changed_list_count": saved_changed_list_count,
            "selected_mapping_count": len(rules),
            "dir_mapping_file": str(dir_mapping_file),
            "mapping_count": len(rules),
            "path_type": path_type,
            "dry_run": dry_run,
            "show_progress": show_progress,
            "num_process": num_process,
            "num_thread": num_thread,
            "moved_count": move_summary["moved_count"],
            "missing_source_count": move_summary["missing_source_count"],
            "conflict_count": move_summary["conflict_count"],
            "moved_samples": move_summary["moved_samples"],
            "missing_samples": move_summary["missing_samples"],
            "conflict_samples": move_summary["conflict_samples"],
            "log_level": log_level,
        }
        logger.log(summary, title="remap dirs mapping-csv summary")
        emit(
            log_level,
            "INFO",
            (
                f"[remap-dirs] mode=mapping_csv path_type={path_type} moved={move_summary['moved_count']}, "
                f"missing={move_summary['missing_source_count']}, conflict={move_summary['conflict_count']}"
            ),
        )
        if out_changed_list_file is not None:
            emit(
                log_level,
                "INFO",
                f"[remap-dirs] out_changed_list_file={out_changed_list_file}",
            )
        return summary

    with list_file.open("r", encoding="utf-8") as f:
        original_rows = [line.rstrip("\n") for line in f]
    tasks = [(idx, row) for idx, row in enumerate(original_rows)]

    if num_process > 1:
        batches = _split_tasks(tasks, num_process)
        with ProcessPoolExecutor(max_workers=num_process) as ex:
            futures = [
                ex.submit(
                    _run_batch,
                    batch,
                    str(dataset_dir),
                    rules,
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
            dataset_dir_str=str(dataset_dir),
            rules=rules,
            skip_missing_files=skip_missing_files,
            dry_run=dry_run,
            num_thread=num_thread,
            process_idx=0,
            show_progress=show_progress,
        )

    results.sort(key=lambda r: int(r["idx"]))
    updated_rows: list[str] = [""] * len(original_rows)
    mapped_line_count = 0
    changed_line_count = 0
    renamed_count = 0
    missing_files: list[str] = []
    conflict_files: list[str] = []
    sample_renames: list[tuple[str, str]] = []

    for r in results:
        idx = int(r["idx"])
        updated_rows[idx] = str(r["updated_row"])
        if bool(r["mapped"]):
            mapped_line_count += 1
        if bool(r["changed_line"]):
            changed_line_count += 1
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
        "out_dataset_dir": str(out_dataset_dir),
        "list_file": str(list_file),
        "out_list_file": str(out_list_file),
        "out_changed_list_file": str(out_changed_list_file) if out_changed_list_file else None,
        "dir_mapping_file": str(dir_mapping_file),
        "mode": "path_list",
        "mapping_count": len(rules),
        "skip_missing_files": skip_missing_files,
        "dry_run": dry_run,
        "show_progress": show_progress,
        "num_process": num_process,
        "num_thread": num_thread,
        "line_count": len(original_rows),
        "mapped_line_count": mapped_line_count,
        "changed_line_count": changed_line_count,
        "renamed_count": renamed_count,
        "missing_file_count": len(missing_files),
        "conflict_count": len(conflict_files),
        "sample_renames": sample_renames,
        "log_level": log_level,
    }
    if out_changed_list_file is not None:
        changed_rows = [updated_rows[int(r["idx"])] for r in results if bool(r["renamed"])]
        out_changed_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_changed_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for rel in changed_rows:
                f.write(rel + "\n")
        summary["saved_changed_list_count"] = len(changed_rows)
    logger.log(summary, title="remap dirs summary")
    emit(
        log_level,
        "INFO",
        (
            f"[remap-dirs] lines={len(original_rows)}, mapped={mapped_line_count}, "
            f"renamed={renamed_count}, missing={len(missing_files)}, conflict={len(conflict_files)}"
        ),
    )
    if out_changed_list_file is not None:
        emit(log_level, "INFO", f"[remap-dirs] out_changed_list_file={out_changed_list_file}")
    return summary

