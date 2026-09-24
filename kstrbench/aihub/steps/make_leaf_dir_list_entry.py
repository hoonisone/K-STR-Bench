from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def _list_leaf_dirs(root_dir: Path, include_root_if_leaf: bool) -> list[Path]:
    leaves: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:
            leaves.append(Path(dirpath).resolve())
    leaves.sort()
    if include_root_if_leaf and not leaves:
        leaves.append(root_dir.resolve())
    return leaves


class MakeLeafDirListStep:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        out_list_file: Any | None = None,
        cache_file: Any | None = None,
        skip_if_exists: Any = False,
        store_relative: Any = True,
        include_root_if_leaf: Any = True,
        log_level: Any | None = None,
        verbose: Any | None = None,
    ) -> None:
        self.dataset_dir_raw = dataset_dir
        self.out_list_file_raw = out_list_file
        self.cache_file_raw = cache_file
        self.skip_if_exists = skip_if_exists
        self.store_relative = store_relative
        self.include_root_if_leaf = include_root_if_leaf
        self.log_level_raw = log_level
        self.verbose = verbose

    def run(self) -> dict[str, Any]:
        if not self.dataset_dir_raw:
            raise ValueError("dataset_dir is required")
        if not self.out_list_file_raw:
            raise ValueError("out_list_file is required")

        dataset_dir = Path(str(self.dataset_dir_raw)).expanduser().resolve()
        out_list_file = Path(str(self.out_list_file_raw)).expanduser().resolve()
        cache_file = (
            Path(str(self.cache_file_raw)).expanduser().resolve()
            if self.cache_file_raw
            else None
        )
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

        if self.log_level_raw is not None:
            log_level = normalize_log_level(self.log_level_raw, default="INFO")
        elif self.verbose is not None:
            # backward compatibility: verbose=true -> INFO, false -> WARNING
            log_level = "INFO" if coerce_bool(self.verbose, default=True) else "WARNING"
        else:
            log_level = "INFO"

        store_relative = coerce_bool(self.store_relative, default=True)
        include_root_if_leaf = coerce_bool(self.include_root_if_leaf, default=True)
        skip_if_exists = coerce_bool(self.skip_if_exists, default=False)

        if skip_if_exists and out_list_file.is_file():
            emit(log_level, "INFO", f"[leaf_dir_list] skip_if_exists=True, reusing: {out_list_file}")
            logger = StepLogger("make_leaf_dir_list_entry")
            logger.log(
                {
                    "dataset_dir": str(dataset_dir),
                    "out_list_file": str(out_list_file),
                    "cache_file": str(cache_file) if cache_file else None,
                    "cache_hit": None,
                    "leaf_dir_count": None,
                    "store_relative": store_relative,
                    "include_root_if_leaf": include_root_if_leaf,
                    "skip_if_exists": skip_if_exists,
                    "skipped": True,
                    "log_level": log_level,
                },
                title="leaf dir list result",
            )
            return {
                "ok": True,
                "dataset_dir": str(dataset_dir),
                "out_list_file": str(out_list_file),
                "cache_file": str(cache_file) if cache_file else None,
                "cache_hit": None,
                "leaf_dir_count": None,
                "store_relative": store_relative,
                "include_root_if_leaf": include_root_if_leaf,
                "skip_if_exists": skip_if_exists,
                "skipped": True,
                "log_level": log_level,
            }

        cache_hit = bool(cache_file and cache_file.is_file())
        lines: list[str]
        if cache_hit:
            with cache_file.open("r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f]
            emit(log_level, "INFO", f"[leaf_dir_list] cache hit: {cache_file}")
        else:
            leaf_dirs = _list_leaf_dirs(dataset_dir, include_root_if_leaf=include_root_if_leaf)
            if store_relative:
                lines = [p.relative_to(dataset_dir).as_posix() for p in leaf_dirs]
            else:
                lines = [str(p) for p in leaf_dirs]
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with cache_file.open("w", encoding="utf-8", newline="\n") as f:
                    for line in lines:
                        f.write(line + "\n")
                emit(log_level, "INFO", f"[leaf_dir_list] cache saved: {cache_file}")

        out_list_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file and cache_file.is_file() and cache_file != out_list_file:
            shutil.copyfile(cache_file, out_list_file)
        else:
            with out_list_file.open("w", encoding="utf-8", newline="\n") as f:
                for line in lines:
                    f.write(line + "\n")

        emit(log_level, "INFO", f"[leaf_dir_list] dataset_dir={dataset_dir}")
        emit(log_level, "INFO", f"[leaf_dir_list] leaf_dir_count={len(lines)}")
        emit(log_level, "INFO", f"[leaf_dir_list] out_list_file={out_list_file}")

        logger = StepLogger("make_leaf_dir_list_entry")
        logger.log(
            {
                "dataset_dir": str(dataset_dir),
                "out_list_file": str(out_list_file),
                "cache_file": str(cache_file) if cache_file else None,
                "cache_hit": cache_hit,
                "leaf_dir_count": len(lines),
                "store_relative": store_relative,
                "include_root_if_leaf": include_root_if_leaf,
                "skip_if_exists": skip_if_exists,
                "skipped": False,
                "log_level": log_level,
            },
            title="leaf dir list result",
        )
        return {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "out_list_file": str(out_list_file),
            "cache_file": str(cache_file) if cache_file else None,
            "cache_hit": cache_hit,
            "leaf_dir_count": len(lines),
            "store_relative": store_relative,
            "include_root_if_leaf": include_root_if_leaf,
            "skip_if_exists": skip_if_exists,
            "skipped": False,
            "log_level": log_level,
        }
