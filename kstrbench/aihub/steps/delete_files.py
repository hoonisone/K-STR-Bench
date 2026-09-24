from __future__ import annotations

import os
import shutil
from typing import Any

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .delete_archives_entry import DeleteArchivesStep
from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class DeleteFiles:
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)

    def run(self) -> dict[str, Any]:
        # backward-compatible path: 기존 archive 삭제 스텝 위임
        if self._kwargs.get("zip_list_file") and not self._kwargs.get("list_file"):
            return DeleteArchivesStep(**self._kwargs).run()
        return run_from_config(self._kwargs)


class DeletePathsFromListStep(DeleteFiles):
    """목록 기반 경로(파일/폴더) 삭제용 명시적 이름 래퍼."""


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


def _parse_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for x in value:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    raise ValueError("filter value must be string or list of strings")


def _normalize_extensions(exts: list[str]) -> set[str]:
    out: set[str] = set()
    for x in exts:
        s = str(x).strip().lower()
        if not s:
            continue
        out.add(s.lstrip("."))
    return out


def _resolve_abs_path(dataset_dir: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p.expanduser().resolve() if p.is_absolute() else (dataset_dir / p).resolve()


def _should_drop(path_text: str, keywords: list[str], ext_set: set[str]) -> bool:
    normalized = path_text.replace("\\", "/")
    if keywords and any(k in normalized for k in keywords):
        return True
    suffix = Path(normalized).suffix.lower().lstrip(".")
    if suffix and suffix in ext_set:
        return True
    return False


def _process_one_row(
    task: tuple[int, str],
    dataset_dir_str: str,
    keywords: list[str],
    ext_set: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    idx, row = task
    trimmed = row.strip()
    if not trimmed:
        return {
            "idx": idx,
            "keep": True,
            "line": "",
            "matched": False,
            "deleted": False,
            "missing": False,
            "error": None,
        }

    matched = _should_drop(trimmed, keywords=keywords, ext_set=ext_set)
    if not matched:
        return {
            "idx": idx,
            "keep": True,
            "line": trimmed,
            "matched": False,
            "deleted": False,
            "missing": False,
            "error": None,
        }

    abs_path = _resolve_abs_path(Path(dataset_dir_str), trimmed)
    if not abs_path.exists():
        return {
            "idx": idx,
            "keep": False,
            "line": trimmed,
            "matched": True,
            "deleted": False,
            "missing": True,
            "error": None,
        }

    if abs_path.is_dir():
        return {
            "idx": idx,
            "keep": False,
            "line": trimmed,
            "matched": True,
            "deleted": False,
            "missing": False,
            "error": f"path is directory, not file: {abs_path}",
        }

    if not dry_run:
        abs_path.unlink()
    return {
        "idx": idx,
        "keep": False,
        "line": trimmed,
        "matched": True,
        "deleted": True,
        "missing": False,
        "error": None,
    }


def _process_delete_target(
    task: tuple[int, str],
    dataset_dir_str: str,
    dry_run: bool,
) -> dict[str, Any]:
    idx, row = task
    trimmed = row.strip()
    if not trimmed:
        return {
            "idx": idx,
            "line": "",
            "deleted": False,
            "missing": False,
            "deleted_kind": None,
            "error": None,
        }

    abs_path = _resolve_abs_path(Path(dataset_dir_str), trimmed)
    if not abs_path.exists():
        return {
            "idx": idx,
            "line": trimmed,
            "deleted": False,
            "missing": True,
            "deleted_kind": None,
            "error": None,
        }
    if abs_path.is_dir():
        if not dry_run:
            shutil.rmtree(abs_path)
        return {
            "idx": idx,
            "line": trimmed,
            "deleted": True,
            "missing": False,
            "deleted_kind": "dir",
            "error": None,
        }

    if not dry_run:
        abs_path.unlink()
    return {
        "idx": idx,
        "line": trimmed,
        "deleted": True,
        "missing": False,
        "deleted_kind": "file",
        "error": None,
    }


def _run_batch(
    batch_tasks: list[tuple[int, str]],
    dataset_dir_str: str,
    keywords: list[str],
    ext_set: set[str],
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
                desc="delete by filter [p0:t0]",
                unit="line",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_tasks
        )
        out: list[dict[str, Any]] = []
        for task in iter_tasks:
            out.append(
                _process_one_row(
                    task=task,
                    dataset_dir_str=dataset_dir_str,
                    keywords=keywords,
                    ext_set=ext_set,
                    dry_run=dry_run,
                )
            )
        return out

    if num_thread <= 1:
        return run_chunk(batch_tasks, thread_idx=0)

    out: list[dict[str, Any]] = []
    chunks = _split_tasks(batch_tasks, num_thread)
    with ThreadPoolExecutor(max_workers=num_thread) as ex:
        futures = [ex.submit(run_chunk, chunk, i) for i, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


def _run_delete_batch(
    batch_tasks: list[tuple[int, str]],
    dataset_dir_str: str,
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
                desc="delete listed paths [p0:t0]",
                unit="line",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_tasks
        )
        out: list[dict[str, Any]] = []
        for task in iter_tasks:
            out.append(
                _process_delete_target(
                    task=task,
                    dataset_dir_str=dataset_dir_str,
                    dry_run=dry_run,
                )
            )
        return out

    if num_thread <= 1:
        return run_chunk(batch_tasks, thread_idx=0)

    out: list[dict[str, Any]] = []
    chunks = _split_tasks(batch_tasks, num_thread)
    with ThreadPoolExecutor(max_workers=num_thread) as ex:
        futures = [ex.submit(run_chunk, chunk, i) for i, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


def _remove_empty_dirs_dir_only(root_dir: Path, dry_run: bool) -> dict[str, Any]:
    # 파일 엔트리는 무시하고 디렉터리만 수집
    all_dirs: list[Path] = []
    stack: list[Path] = [root_dir]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                subdirs: list[Path] = []
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        sub = Path(entry.path)
                        subdirs.append(sub)
                if subdirs:
                    all_dirs.extend(subdirs)
                    stack.extend(subdirs)
        except FileNotFoundError:
            continue

    # 하위 폴더부터 삭제 시도
    all_dirs.sort(key=lambda p: len(p.parts), reverse=True)
    removed_count = 0
    for d in all_dirs:
        try:
            with os.scandir(d) as it:
                has_any_entry = next(it, None) is not None
            if has_any_entry:
                continue
            if not dry_run:
                d.rmdir()
            removed_count += 1
        except FileNotFoundError:
            continue
        except OSError:
            # 권한/경합 등은 스킵
            continue

    return {
        "removed_empty_dir_count": removed_count,
        "scanned_dir_count": len(all_dirs),
        "dry_run": dry_run,
        "strategy": "dir_only_scan",
    }


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir")
    if not dataset_dir_raw:
        raise ValueError("dataset_dir is required")
    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    delete_list_file_raw = params.get("delete_list_file", params.get("target_list_file"))
    list_file_raw = params.get("list_file")
    if delete_list_file_raw:
        list_file = Path(str(delete_list_file_raw)).expanduser().resolve()
    else:
        if not list_file_raw:
            raise ValueError("list_file is required")
        list_file = Path(str(list_file_raw)).expanduser().resolve()
    if not list_file.is_file():
        raise FileNotFoundError(f"list_file not found: {list_file}")

    out_list_file_raw = params.get("out_list_file")
    out_list_file = (
        Path(str(out_list_file_raw)).expanduser().resolve()
        if out_list_file_raw
        else list_file
    )

    direct_delete_mode = bool(delete_list_file_raw)
    keywords = _parse_text_list(params.get("drop_keywords", params.get("keywords")))
    remove_extensions = _parse_text_list(
        params.get("drop_extensions", params.get("extensions"))
    )
    ext_set = _normalize_extensions(remove_extensions)
    if not direct_delete_mode and not keywords and not ext_set:
        raise ValueError(
            "drop_keywords/drop_extensions is required "
            "(or use delete_list_file for direct delete mode)"
        )

    if params.get("log_level") is not None:
        log_level = normalize_log_level(params.get("log_level"), default="INFO")
    elif params.get("verbose") is not None:
        log_level = "INFO" if coerce_bool(params.get("verbose"), default=True) else "WARNING"
    else:
        log_level = "INFO"

    dry_run = coerce_bool(params.get("dry_run", False), default=False)
    show_progress = coerce_bool(params.get("show_progress", True), default=True)
    remove_empty_folder = coerce_bool(params.get("remove_empty_folder", False), default=False)
    num_process, num_thread = _parse_parallelism(params)

    with list_file.open("r", encoding="utf-8") as f:
        original_rows = [line.rstrip("\n") for line in f]

    logger = StepLogger("delete_files")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "list_file": str(list_file),
            "out_list_file": str(out_list_file),
            "direct_delete_mode": direct_delete_mode,
            "drop_keywords": keywords,
            "drop_extensions": sorted(ext_set),
            "dry_run": dry_run,
            "show_progress": show_progress,
            "remove_empty_folder": remove_empty_folder,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
            "line_count": len(original_rows),
        },
        title="delete discovery",
    )
    emit(
        log_level,
        "INFO",
        f"[delete-files] lines={len(original_rows)}, direct_delete_mode={direct_delete_mode}",
    )

    tasks = [(idx, row) for idx, row in enumerate(original_rows)]
    dataset_dir_str = str(dataset_dir)

    if direct_delete_mode:
        if num_process > 1:
            batches = _split_tasks(tasks, num_process)
            with ProcessPoolExecutor(max_workers=num_process) as ex:
                futures = [
                    ex.submit(
                        _run_delete_batch,
                        batch,
                        dataset_dir_str,
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
            results = _run_delete_batch(
                batch_tasks=tasks,
                dataset_dir_str=dataset_dir_str,
                dry_run=dry_run,
                num_thread=num_thread,
                process_idx=0,
                show_progress=show_progress,
            )
    else:
        if num_process > 1:
            batches = _split_tasks(tasks, num_process)
            with ProcessPoolExecutor(max_workers=num_process) as ex:
                futures = [
                    ex.submit(
                        _run_batch,
                        batch,
                        dataset_dir_str,
                        keywords,
                        ext_set,
                        dry_run,
                        num_thread,
                        process_idx,
                        show_progress,
                    )
                    for process_idx, batch in enumerate(batches)
                ]
                results = []
                for fut in as_completed(futures):
                    results.extend(fut.result())
        else:
            results = _run_batch(
                batch_tasks=tasks,
                dataset_dir_str=dataset_dir_str,
                keywords=keywords,
                ext_set=ext_set,
                dry_run=dry_run,
                num_thread=num_thread,
                process_idx=0,
                show_progress=show_progress,
            )

    results.sort(key=lambda r: int(r["idx"]))
    kept_rows: list[str] = []
    matched_count = 0
    deleted_count = 0
    deleted_file_count = 0
    deleted_dir_count = 0
    missing_count = 0
    error_rows: list[str] = []
    removed_samples: list[str] = []
    missing_samples: list[str] = []

    for r in results:
        if bool(r["deleted"]):
            deleted_count += 1
            if r.get("deleted_kind") == "file":
                deleted_file_count += 1
            elif r.get("deleted_kind") == "dir":
                deleted_dir_count += 1
            if r["line"] and len(removed_samples) < 10:
                removed_samples.append(str(r["line"]))
        if bool(r["missing"]):
            missing_count += 1
            if r["line"] and len(missing_samples) < 10:
                missing_samples.append(str(r["line"]))
        if (not direct_delete_mode) and bool(r["matched"]):
            matched_count += 1
        if r["error"]:
            error_rows.append(str(r["error"]))
        if (not direct_delete_mode) and bool(r["keep"]):
            kept_rows.append(str(r["line"]))

    if (not direct_delete_mode) and (not dry_run):
        out_list_file.parent.mkdir(parents=True, exist_ok=True)
        with out_list_file.open("w", encoding="utf-8", newline="\n") as f:
            for row in kept_rows:
                f.write(row + "\n")

    empty_summary: dict[str, Any] = {"skipped": True, "removed_empty_dir_count": 0}
    if remove_empty_folder:
        empty_summary = _remove_empty_dirs_dir_only(dataset_dir, dry_run=dry_run)

    summary = {
        "ok": len(error_rows) == 0,
        "dataset_dir": str(dataset_dir),
        "list_file": str(list_file),
        "out_list_file": str(out_list_file),
        "direct_delete_mode": direct_delete_mode,
        "drop_keywords": keywords,
        "drop_extensions": sorted(ext_set),
        "dry_run": dry_run,
        "show_progress": show_progress,
        "remove_empty_folder": remove_empty_folder,
        "num_process": num_process,
        "num_thread": num_thread,
        "line_count": len(original_rows),
        "kept_line_count": len(kept_rows) if not direct_delete_mode else None,
        "matched_count": matched_count,
        "deleted_count": deleted_count,
        "deleted_file_count": deleted_file_count,
        "deleted_dir_count": deleted_dir_count,
        "missing_count": missing_count,
        "error_count": len(error_rows),
        "removed_samples": removed_samples,
        "missing_samples": missing_samples,
        "remove_empty_dirs_summary": empty_summary,
    }
    if error_rows:
        summary["errors"] = error_rows[:20]

    logger.log(summary, title="delete summary")
    emit(
        log_level,
        "INFO",
        (
            f"[delete-files] done matched={matched_count}, deleted={deleted_count}, "
            f"kept={len(kept_rows)}, missing={missing_count}, errors={len(error_rows)}"
        ),
    )
    return summary
