from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
from typing import Any
import zipfile

from .step_logger import emit


def load_rel_paths_from_file(list_file: Path) -> list[str]:
    if not list_file.is_file():
        raise FileNotFoundError(f"zip list file not found: {list_file}")
    rel_paths: list[str] = []
    with list_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                rel_paths.append(line.replace("\\", "/"))
    return rel_paths


def to_rel_path(dataset_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_zip_path(dataset_dir: Path, rel_or_abs: str) -> Path:
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        return candidate.expanduser().resolve()
    return (dataset_dir / candidate).expanduser().resolve()


def find_zip_paths(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.rglob("*.zip"))


def build_compare_summary(dataset_dir: Path, listed_rel_paths: list[str]) -> dict[str, Any]:
    current_rel_paths = sorted(to_rel_path(dataset_dir, p) for p in find_zip_paths(dataset_dir))
    current_set = set(current_rel_paths)
    expected_set = set(listed_rel_paths)
    missing_in_current = sorted(expected_set - current_set)
    extra_in_current = sorted(current_set - expected_set)
    is_match = not missing_in_current and not extra_in_current
    return {
        "expected_count": len(listed_rel_paths),
        "current_count": len(current_rel_paths),
        "is_match": is_match,
        "missing_in_current_count": len(missing_in_current),
        "extra_in_current_count": len(extra_in_current),
        "missing_in_current": missing_in_current,
        "extra_in_current": extra_in_current,
    }


def _split_tasks(tasks: list[tuple[str, str]], num_batches: int) -> list[list[tuple[str, str]]]:
    if not tasks:
        return []
    n = len(tasks)
    num_batches = max(1, min(num_batches, n))
    q, r = divmod(n, num_batches)
    batches: list[list[tuple[str, str]]] = []
    start = 0
    for i in range(num_batches):
        size = q + (1 if i < r else 0)
        end = start + size
        if start < end:
            batches.append(tasks[start:end])
        start = end
    return batches


def _target_dir_for_zip(dataset_dir: Path, origin_path: Path, source_base_dir: Path) -> Path:
    try:
        rel_parent = origin_path.relative_to(source_base_dir).parent
        return dataset_dir / rel_parent / origin_path.stem
    except ValueError:
        return origin_path.parent / origin_path.stem


def build_unzip_tasks(
    dataset_dir: Path,
    zip_paths: list[Path],
    source_base_dir: Path | None = None,
) -> list[tuple[str, str]]:
    source_base = source_base_dir or dataset_dir
    tasks: list[tuple[str, str]] = []
    for origin_path in zip_paths:
        target_dir = _target_dir_for_zip(dataset_dir, origin_path, source_base)
        tasks.append((str(origin_path), str(target_dir)))
    return tasks


def count_existing_targets(tasks: list[tuple[str, str]]) -> int:
    count = 0
    for _, target_dir_str in tasks:
        if Path(target_dir_str).exists():
            count += 1
    return count


def _run_unzip_command(origin_path: Path, target_dir: Path, log_level: str) -> bool:
    unzip_bin = shutil.which("unzip")
    if not unzip_bin:
        return False

    commands = [
        [unzip_bin, "-o", "-O", "cp949", str(origin_path), "-d", str(target_dir)],
        [unzip_bin, "-o", str(origin_path), "-d", str(target_dir)],
    ]
    for cmd in commands:
        emit(log_level, "DEBUG", f"[unzip_cli] {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return True
    return False


def _decode_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def _safe_join(base_dir: Path, relative_name: str) -> Path:
    dest = (base_dir / relative_name).resolve()
    base = base_dir.resolve()
    if dest != base and base not in dest.parents:
        raise ValueError(f"unsafe zip path detected: {relative_name}")
    return dest


def _unzip_with_zipfile(origin_path: Path, target_dir: Path, log_level: str) -> bool:
    try:
        with zipfile.ZipFile(origin_path, "r") as zf:
            for info in zf.infolist():
                member_name = _decode_zip_name(info)
                target_path = _safe_join(target_dir, member_name)
                if info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return True
    except Exception as exc:
        emit(log_level, "ERROR", f"[unzip_zipfile] failed: {origin_path} ({exc})")
        return False


def _unzip_one(origin_path: Path, target_dir: Path, dry_run: bool, log_level: str) -> bool:
    emit(log_level, "INFO", f"[unzip] {origin_path} -> {target_dir}")
    if dry_run:
        return True
    target_dir.mkdir(parents=True, exist_ok=True)
    if _run_unzip_command(origin_path, target_dir, log_level):
        return True
    emit(log_level, "WARNING", f"[unzip] cli unavailable/failed; fallback to zipfile: {origin_path}")
    return _unzip_with_zipfile(origin_path, target_dir, log_level)


def _remove_target_path(target_dir: Path, dry_run: bool, log_level: str) -> bool:
    emit(log_level, "WARNING", f"[unzip] overwrite target: {target_dir}")
    if dry_run:
        return True
    try:
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink(missing_ok=True)
        return True
    except Exception as exc:
        emit(log_level, "ERROR", f"[unzip] failed to remove target {target_dir}: {exc}")
        return False


def _run_unzip_chunk(
    chunk_tasks: list[tuple[str, str]],
    dry_run: bool,
    log_level: str,
    overwrite: bool,
) -> dict[str, int]:
    unzipped_count = 0
    skipped_count = 0
    overwritten_count = 0
    failed_count = 0
    for origin_path_str, target_dir_str in chunk_tasks:
        origin_path = Path(origin_path_str)
        target_dir = Path(target_dir_str)
        if target_dir.exists():
            if overwrite:
                if not _remove_target_path(target_dir, dry_run=dry_run, log_level=log_level):
                    failed_count += 1
                    continue
                overwritten_count += 1
            else:
                skipped_count += 1
                continue
        ok = _unzip_one(origin_path, target_dir, dry_run=dry_run, log_level=log_level)
        if ok:
            unzipped_count += 1
        else:
            failed_count += 1
    return {
        "unzipped_count": unzipped_count,
        "skipped_count": skipped_count,
        "overwritten_count": overwritten_count,
        "failed_count": failed_count,
        "total_zip_count": len(chunk_tasks),
    }


def _run_unzip_batch(
    batch_tasks: list[tuple[str, str]],
    dry_run: bool,
    num_thread: int,
    log_level: str,
    overwrite: bool,
) -> dict[str, int]:
    if num_thread <= 1:
        return _run_unzip_chunk(
            batch_tasks,
            dry_run=dry_run,
            log_level=log_level,
            overwrite=overwrite,
        )

    total = {
        "unzipped_count": 0,
        "skipped_count": 0,
        "overwritten_count": 0,
        "failed_count": 0,
        "total_zip_count": len(batch_tasks),
    }
    chunks = _split_tasks(batch_tasks, num_thread)
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = [
            executor.submit(_run_unzip_chunk, chunk, dry_run, log_level, overwrite)
            for chunk in chunks
        ]
        for fut in as_completed(futures):
            stats = fut.result()
            total["unzipped_count"] += stats["unzipped_count"]
            total["skipped_count"] += stats["skipped_count"]
            total["overwritten_count"] += stats["overwritten_count"]
            total["failed_count"] += stats["failed_count"]
    return total


def unzip_tasks(
    tasks: list[tuple[str, str]],
    dry_run: bool,
    log_level: str,
    num_process: int,
    num_thread: int,
    overwrite: bool = False,
) -> dict[str, int]:
    if num_process <= 1:
        result = _run_unzip_batch(
            tasks,
            dry_run=dry_run,
            num_thread=num_thread,
            log_level=log_level,
            overwrite=overwrite,
        )
        result["dry_run"] = dry_run
        result["overwrite"] = overwrite
        return result

    total = {
        "unzipped_count": 0,
        "skipped_count": 0,
        "overwritten_count": 0,
        "failed_count": 0,
        "total_zip_count": len(tasks),
    }
    batches = _split_tasks(tasks, num_process)
    with ProcessPoolExecutor(max_workers=num_process) as executor:
        futures = [
            executor.submit(_run_unzip_batch, batch, dry_run, num_thread, log_level, overwrite)
            for batch in batches
        ]
        for fut in as_completed(futures):
            stats = fut.result()
            total["unzipped_count"] += stats["unzipped_count"]
            total["skipped_count"] += stats["skipped_count"]
            total["overwritten_count"] += stats["overwritten_count"]
            total["failed_count"] += stats["failed_count"]
    total["dry_run"] = dry_run
    total["overwrite"] = overwrite
    return total


def delete_archives(zip_paths: list[Path], dry_run: bool) -> dict[str, int | bool]:
    deleted_archive_count = 0
    delete_skipped_count = 0
    for p in zip_paths:
        if dry_run:
            if p.exists():
                deleted_archive_count += 1
            continue
        if not p.exists():
            delete_skipped_count += 1
            continue
        p.unlink()
        deleted_archive_count += 1
    return {
        "dry_run": dry_run,
        "deleted_archive_count": deleted_archive_count,
        "delete_skipped_count": delete_skipped_count,
    }


def remove_empty_dirs(root_dir: Path, dry_run: bool) -> dict[str, int | bool]:
    dirs = sorted(
        [p for p in root_dir.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    removed_count = 0
    for d in dirs:
        try:
            next(d.iterdir())
            continue
        except StopIteration:
            if not dry_run:
                d.rmdir()
            removed_count += 1
    return {
        "removed_empty_dir_count": removed_count,
        "dry_run": dry_run,
    }
