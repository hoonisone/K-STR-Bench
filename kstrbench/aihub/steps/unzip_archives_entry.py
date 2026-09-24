from __future__ import annotations

from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level
from ..helpers.unzip_core import (
    build_unzip_tasks,
    count_existing_targets,
    load_rel_paths_from_file,
    resolve_zip_path,
    unzip_tasks,
)


class UnzipArchivesStep:
    def __init__(
        self,
        out_dataset_dir: Any | None = None,
        dataset_dir: Any | None = None,
        source_dataset_dir: Any | None = None,
        zip_list_valid_file: Any | None = None,
        zip_list_file: Any | None = None,
        dry_run: Any = False,
        overwrite: Any = False,
        num_process: Any = 1,
        num_thread: Any = 1,
        log_level: Any | None = None,
        verbose: Any | None = None,
    ) -> None:
        self.out_dataset_dir_raw = out_dataset_dir
        self.dataset_dir_raw = dataset_dir
        self.source_dataset_dir_raw = source_dataset_dir
        self.zip_list_valid_file_raw = zip_list_valid_file
        self.zip_list_file_raw = zip_list_file
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.num_process = num_process
        self.num_thread = num_thread
        self.log_level_raw = log_level
        self.verbose = verbose

    def run(self) -> dict[str, Any]:
        out_dataset_dir_raw = (
            self.out_dataset_dir_raw
            if self.out_dataset_dir_raw is not None
            else self.dataset_dir_raw
        )
        if not out_dataset_dir_raw:
            raise ValueError("out_dataset_dir is required")
        zip_list_valid_file_raw = (
            self.zip_list_valid_file_raw
            if self.zip_list_valid_file_raw is not None
            else self.zip_list_file_raw
        )
        if not zip_list_valid_file_raw:
            raise ValueError("zip_list_valid_file is required")

        out_dataset_dir = Path(str(out_dataset_dir_raw)).expanduser().resolve()
        source_dataset_dir = (
            Path(str(self.source_dataset_dir_raw)).expanduser().resolve()
            if self.source_dataset_dir_raw
            else out_dataset_dir
        )
        zip_list_file = Path(str(zip_list_valid_file_raw)).expanduser().resolve()
        if not source_dataset_dir.is_dir():
            raise FileNotFoundError(f"source_dataset_dir not found: {source_dataset_dir}")
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        if self.log_level_raw is not None:
            log_level = normalize_log_level(self.log_level_raw, default="INFO")
        elif self.verbose is not None:
            log_level = "INFO" if coerce_bool(self.verbose, default=True) else "WARNING"
        else:
            log_level = "INFO"

        dry_run = coerce_bool(self.dry_run, default=False)
        overwrite = coerce_bool(self.overwrite, default=False)
        num_process = int(self.num_process)
        num_thread = int(self.num_thread)
        if num_process < 1:
            raise ValueError("num_process must be >= 1")
        if num_thread < 1:
            raise ValueError("num_thread must be >= 1")

        listed_rel_paths = load_rel_paths_from_file(zip_list_file)
        listed_zip_paths = [resolve_zip_path(source_dataset_dir, rel) for rel in listed_rel_paths]
        missing_zip_count = sum(1 for p in listed_zip_paths if not p.is_file())
        zip_paths = [p for p in listed_zip_paths if p.is_file()]
        tasks = build_unzip_tasks(
            out_dataset_dir,
            zip_paths,
            source_base_dir=source_dataset_dir,
        )
        preexisting_target_count = count_existing_targets(tasks)

        logger = StepLogger("unzip_archives_entry")
        logger.log(
            {
                "out_dataset_dir": str(out_dataset_dir),
                "source_dataset_dir": str(source_dataset_dir),
                "zip_list_valid_file": str(zip_list_file),
                "zip_count": len(zip_paths),
                "missing_zip_count": missing_zip_count,
                "preexisting_target_count": preexisting_target_count,
                "dry_run": dry_run,
                "overwrite": overwrite,
                "num_process": num_process,
                "num_thread": num_thread,
                "log_level": log_level,
            },
            title="unzip discovery",
        )
        emit(log_level, "INFO", f"[unzip] zip_count={len(zip_paths)}, missing={missing_zip_count}")
        emit(log_level, "INFO", f"[unzip] workers process={num_process}, thread={num_thread}, overwrite={overwrite}")

        unzip_summary = unzip_tasks(
            tasks=tasks,
            dry_run=dry_run,
            log_level=log_level,
            num_process=num_process,
            num_thread=num_thread,
            overwrite=overwrite,
        )
        unzip_summary["preexisting_target_count"] = preexisting_target_count
        logger.log(unzip_summary, title="unzip summary")
        return {
            "ok": True,
            "out_dataset_dir": str(out_dataset_dir),
            "dataset_dir": str(out_dataset_dir),
            "source_dataset_dir": str(source_dataset_dir),
            "zip_list_valid_file": str(zip_list_file),
            "zip_count": len(zip_paths),
            "missing_zip_count": missing_zip_count,
            "dry_run": dry_run,
            "overwrite": overwrite,
            "num_process": num_process,
            "num_thread": num_thread,
            "log_level": log_level,
            "unzip_summary": unzip_summary,
        }


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return UnzipArchivesStep(**dict(params or {})).run()
