from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy file/folder with process+thread workers.")
    parser.add_argument("src", type=Path, help="Source file or folder path")
    parser.add_argument("dst", type=Path, help="Destination file or folder path")
    parser.add_argument(
        "--num-process",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of worker threads per process (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist at destination (default behavior)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress from first process/thread only",
    )
    return parser.parse_args()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(value)


def _split_tasks(
    tasks: list[tuple[str, str, int]],
    num_batches: int,
) -> list[list[tuple[str, str, int]]]:
    if not tasks:
        return []
    n = len(tasks)
    num_batches = max(1, min(num_batches, n))
    q, r = divmod(n, num_batches)
    batches: list[list[tuple[str, str, int]]] = []
    start = 0
    for i in range(num_batches):
        size = q + (1 if i < r else 0)
        end = start + size
        if start < end:
            batches.append(tasks[start:end])
        start = end
    return batches


def _build_copy_tasks(src: Path, dst: Path, show_progress: bool) -> list[tuple[str, str, int]]:
    tasks: list[tuple[str, str, int]] = []
    if src.is_file():
        if dst.exists() and dst.is_dir():
            dst_file = dst / src.name
        else:
            dst_file = dst
        size = int(src.stat().st_size)
        tasks.append((str(src), str(dst_file), size))
        return tasks

    if not src.is_dir():
        raise ValueError(f"Source path is not file/dir: {src}")

    path_iter: Any = (
        tqdm(src.rglob("*"), desc="scan source", unit="path", dynamic_ncols=True)
        if show_progress
        else src.rglob("*")
    )
    for p in path_iter:
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        dst_file = dst / rel
        tasks.append((str(p), str(dst_file), int(p.stat().st_size)))
    return tasks


def _normalize_operation(raw: Any, default: str = "move") -> str:
    op = str(raw if raw is not None else default).strip().lower()
    if op not in ("move", "copy"):
        raise ValueError("operation must be 'move' or 'copy'")
    return op


def _transfer_one_file(
    src_file: str,
    dst_file: str,
    overwrite: bool,
    operation: str,
) -> tuple[int, int]:
    src = Path(src_file)
    dst = Path(dst_file)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return (0, 1)
    size = int(src.stat().st_size)
    if dst.exists():
        dst.unlink()
    if operation == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)
    return (size, 0)


def _run_batch(
    batch_tasks: list[tuple[str, str, int]],
    overwrite: bool,
    num_thread: int,
    process_idx: int,
    show_progress: bool,
    operation: str,
) -> tuple[int, int, int]:
    def run_chunk(chunk_tasks: list[tuple[str, str, int]], thread_idx: int) -> tuple[int, int, int]:
        copied_bytes = 0
        copied_files = 0
        skipped_files = 0
        iter_tasks: Any = (
            tqdm(
                chunk_tasks,
                desc=f"{operation} [p0:t0]",
                unit="file",
                dynamic_ncols=True,
            )
            if show_progress and process_idx == 0 and thread_idx == 0
            else chunk_tasks
        )
        for src_file, dst_file, _ in iter_tasks:
            size, skipped = _transfer_one_file(src_file, dst_file, overwrite, operation)
            copied_bytes += size
            if skipped:
                skipped_files += 1
            else:
                copied_files += 1
        return copied_bytes, copied_files, skipped_files

    if num_thread <= 1:
        return run_chunk(batch_tasks, thread_idx=0)

    chunks = _split_tasks(batch_tasks, num_thread)
    total_bytes = 0
    total_copied = 0
    total_skipped = 0
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = [
            executor.submit(run_chunk, chunk, thread_idx)
            for thread_idx, chunk in enumerate(chunks)
        ]
        for fut in as_completed(futures):
            b, c, s = fut.result()
            total_bytes += b
            total_copied += c
            total_skipped += s
    return total_bytes, total_copied, total_skipped


def _remove_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def _move_whole_path(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def copy_path(
    src: Path,
    dst: Path,
    num_process: int,
    num_thread: int,
    overwrite: bool,
    show_progress: bool,
    operation: str = "move",
) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    if num_process < 1:
        raise ValueError("num_process must be >= 1 (use 1 to disable multiprocessing)")
    if num_thread < 1:
        raise ValueError("num_thread must be >= 1 (use 1 to disable multithreading)")

    if operation == "move":
        start = time.perf_counter()
        src_is_dir = src.is_dir()
        _move_whole_path(src, dst, overwrite=overwrite)
        elapsed = time.perf_counter() - start
        return {
            "ok": True,
            "source": str(src),
            "destination": str(dst),
            "operation": operation,
            "moved_as": "dir" if src_is_dir else "file",
            "num_process": num_process,
            "num_thread": num_thread,
            "elapsed_sec": round(elapsed, 3),
        }

    scan_t0 = time.perf_counter()
    tasks = _build_copy_tasks(src, dst, show_progress=show_progress)
    scan_elapsed = time.perf_counter() - scan_t0

    total_files = len(tasks)
    total_bytes = sum(size for _, _, size in tasks)
    if total_files == 0:
        dst.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "source": str(src),
            "destination": str(dst),
            "num_process": num_process,
            "num_thread": num_thread,
            "total_files": 0,
            "copied_files": 0,
            "skipped_files": 0,
            "copied_bytes": 0,
            "source_bytes": 0,
            "scan_elapsed_sec": round(scan_elapsed, 3),
            "elapsed_sec": 0.0,
            "avg_speed_mb_s": 0.0,
        }

    start = time.perf_counter()
    copied_bytes = 0
    copied_files = 0
    skipped_files = 0
    if num_process > 1:
        batches = _split_tasks(tasks, num_process)
        with ProcessPoolExecutor(max_workers=num_process) as executor:
            futures = [
                executor.submit(
                    _run_batch,
                    batch,
                    overwrite,
                    num_thread,
                    p_idx,
                    show_progress,
                    operation,
                )
                for p_idx, batch in enumerate(batches)
            ]
            for future in as_completed(futures):
                b, c, s = future.result()
                copied_bytes += b
                copied_files += c
                skipped_files += s
    else:
        b, c, s = _run_batch(tasks, overwrite, num_thread, 0, show_progress, operation)
        copied_bytes += b
        copied_files += c
        skipped_files += s

    if operation == "move" and src.is_dir():
        _remove_empty_dirs(src)

    elapsed = time.perf_counter() - start
    mb = copied_bytes / (1024 * 1024)
    speed = mb / elapsed if elapsed > 0 else 0.0
    return {
        "ok": True,
        "source": str(src),
        "destination": str(dst),
        "operation": operation,
        "num_process": num_process,
        "num_thread": num_thread,
        "total_files": total_files,
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "copied_bytes": copied_bytes,
        "source_bytes": total_bytes,
        "scan_elapsed_sec": round(scan_elapsed, 3),
        "elapsed_sec": round(elapsed, 3),
        "avg_speed_mb_s": round(speed, 3),
    }


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    src_raw = params.get("src")
    dst_raw = params.get("dst")
    if not src_raw:
        raise ValueError("params.src is required")
    if not dst_raw:
        raise ValueError("params.dst is required")

    src = Path(str(src_raw)).expanduser().resolve()
    dst = Path(str(dst_raw)).expanduser().resolve()
    num_process = int(params.get("num_process", params.get("num_proc", 1)))
    num_thread = int(
        params.get(
            "num_thread",
            params.get("num_threads", params.get("num_worker", 1)),
        )
    )
    overwrite = _coerce_bool(params.get("overwrite", False), default=False)
    show_progress = _coerce_bool(params.get("show_progress", True), default=True)
    operation = _normalize_operation(params.get("operation"), default="move")
    return copy_path(
        src=src,
        dst=dst,
        num_process=num_process,
        num_thread=num_thread,
        overwrite=overwrite,
        show_progress=show_progress,
        operation=operation,
    )


def main() -> None:
    args = parse_args()
    if args.overwrite and args.skip_existing:
        raise ValueError("Use only one option: --overwrite or --skip-existing")
    overwrite = args.overwrite and not args.skip_existing
    res = copy_path(
        src=args.src.expanduser().resolve(),
        dst=args.dst.expanduser().resolve(),
        num_process=args.num_process,
        num_thread=args.num_thread,
        overwrite=overwrite,
        show_progress=args.show_progress,
    )
    print(f"Source      : {res['source']}")
    print(f"Destination : {res['destination']}")
    print(f"Processes   : {res['num_process']}")
    print(f"Threads     : {res['num_thread']}")
    print(f"Total files : {res['total_files']}")
    print(f"Copied files: {res['copied_files']}")
    print(f"Skipped     : {res['skipped_files']}")
    print(f"Copied size : {res['copied_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Scan        : {res['scan_elapsed_sec']:.2f} sec")
    print(f"Elapsed     : {res['elapsed_sec']:.2f} sec")
    print(f"Avg speed   : {res['avg_speed_mb_s']:.2f} MB/s")


if __name__ == "__main__":
    main()
