from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _normalize_ext(ext: str) -> str:
    e = ext.strip()
    if not e:
        return e
    return e if e.startswith(".") else f".{e}"


def _scan_files_under(
    root_abs: Path,
    exts: set[str] | None,
) -> list[Path]:
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root_abs):
        dp = Path(dirpath)
        for name in filenames:
            p = dp / name
            if exts is None or p.suffix in exts:
                out.append(p.resolve())
    return out


class MakePathLIst:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})

    dataset_dir_raw = params.get("dataset_dir", params.get("datadir"))
    scan_roots_file_raw = params.get("scan_roots_file")
    out_list_file_raw = params.get("out_list_file")
    cache_file_raw = params.get("cache_file")
    extensions_raw = params.get("extensions")
    store_relative = coerce_bool(params.get("store_relative", True), default=True)
    skip_if_exists = coerce_bool(params.get("skip_if_exists", False), default=False)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not dataset_dir_raw:
        raise ValueError("dataset_dir (or datadir) is required")
    if not scan_roots_file_raw:
        raise ValueError("scan_roots_file is required")
    if not out_list_file_raw:
        raise ValueError("out_list_file is required")

    dataset_dir = Path(str(dataset_dir_raw)).expanduser().resolve()
    scan_roots_file = Path(str(scan_roots_file_raw)).expanduser().resolve()
    out_list_file = Path(str(out_list_file_raw)).expanduser().resolve()
    cache_file = (
        Path(str(cache_file_raw)).expanduser().resolve() if cache_file_raw else None
    )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    if not scan_roots_file.is_file():
        raise FileNotFoundError(f"scan_roots_file not found: {scan_roots_file}")

    logger = StepLogger("make_path_list")
    logger.log(
        {
            "dataset_dir": str(dataset_dir),
            "scan_roots_file": str(scan_roots_file),
            "out_list_file": str(out_list_file),
            "cache_file": str(cache_file) if cache_file else None,
            "extensions": extensions_raw,
            "store_relative": store_relative,
            "skip_if_exists": skip_if_exists,
            "log_level": log_level,
        },
        title="make path list discovery",
    )

    if skip_if_exists and out_list_file.is_file() and out_list_file.stat().st_size > 0:
        emit(log_level, "INFO", f"[make_path_list] skip_if_exists=True, reuse: {out_list_file}")
        return {
            "ok": True,
            "skipped": True,
            "out_list_file": str(out_list_file),
            "log_level": log_level,
        }

    if cache_file and cache_file.is_file():
        out_list_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("r", encoding="utf-8") as src, out_list_file.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            data = src.read()
            dst.write(data)
        lines = [ln for ln in data.splitlines() if ln.strip()]
        emit(log_level, "INFO", f"[make_path_list] cache hit: {cache_file}")
        return {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "scan_roots_file": str(scan_roots_file),
            "out_list_file": str(out_list_file),
            "cache_file": str(cache_file),
            "path_count": len(lines),
            "store_relative": store_relative,
            "skipped": False,
            "log_level": log_level,
        }

    exts: set[str] | None = None
    if isinstance(extensions_raw, (list, tuple)) and len(extensions_raw) > 0:
        exts = {_normalize_ext(str(e)) for e in extensions_raw if str(e).strip()}

    roots: list[Path] = []
    with scan_roots_file.open("r", encoding="utf-8") as f:
        for raw in f:
            rel = raw.strip().replace("\\", "/")
            if not rel:
                continue
            abs_root = (dataset_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
            if abs_root.is_dir():
                roots.append(abs_root)

    files: list[Path] = []
    for root in roots:
        files.extend(_scan_files_under(root, exts=exts))
    files = sorted(set(files))

    lines: list[str] = []
    for p in files:
        if store_relative:
            try:
                lines.append(p.relative_to(dataset_dir).as_posix())
            except ValueError:
                lines.append(str(p))
        else:
            lines.append(str(p))

    out_list_file.parent.mkdir(parents=True, exist_ok=True)
    with out_list_file.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")

    result = {
        "ok": True,
        "dataset_dir": str(dataset_dir),
        "scan_roots_file": str(scan_roots_file),
        "out_list_file": str(out_list_file),
        "cache_file": str(cache_file) if cache_file else None,
        "path_count": len(lines),
        "store_relative": store_relative,
        "skipped": False,
        "log_level": log_level,
    }
    logger.log(result, title="make path list summary")
    emit(log_level, "INFO", f"[make_path_list] paths={len(lines)} -> {out_list_file}")
    return result
