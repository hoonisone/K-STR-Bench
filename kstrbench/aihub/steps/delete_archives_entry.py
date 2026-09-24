from __future__ import annotations

from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level
from ..helpers.unzip_core import delete_archives, load_rel_paths_from_file, remove_empty_dirs, resolve_zip_path


class DeleteArchivesStep:
    def __init__(
        self,
        dataset_dir: Any | None = None,
        zip_list_file: Any | None = None,
        delete_archive: Any = True,
        remove_empty_folder: Any = True,
        dry_run: Any = False,
        log_level: Any | None = None,
        verbose: Any | None = None,
    ) -> None:
        self.dataset_dir_raw = dataset_dir
        self.zip_list_file_raw = zip_list_file
        self.delete_archive = delete_archive
        self.remove_empty_folder = remove_empty_folder
        self.dry_run = dry_run
        self.log_level_raw = log_level
        self.verbose = verbose

    def run(self) -> dict[str, Any]:
        if not self.dataset_dir_raw:
            raise ValueError("dataset_dir is required")
        if not self.zip_list_file_raw:
            raise ValueError("zip_list_file is required")

        dataset_dir = Path(str(self.dataset_dir_raw)).expanduser().resolve()
        zip_list_file = Path(str(self.zip_list_file_raw)).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

        if self.log_level_raw is not None:
            log_level = normalize_log_level(self.log_level_raw, default="INFO")
        elif self.verbose is not None:
            log_level = "INFO" if coerce_bool(self.verbose, default=True) else "WARNING"
        else:
            log_level = "INFO"

        delete_archive_flag = coerce_bool(self.delete_archive, default=True)
        remove_empty_folder = coerce_bool(self.remove_empty_folder, default=True)
        dry_run = coerce_bool(self.dry_run, default=False)

        listed_rel_paths = load_rel_paths_from_file(zip_list_file)
        listed_zip_paths = [resolve_zip_path(dataset_dir, rel) for rel in listed_rel_paths]

        logger = StepLogger("delete_archives_entry")
        delete_summary: dict[str, Any]
        if delete_archive_flag:
            delete_summary = delete_archives(listed_zip_paths, dry_run=dry_run)
        else:
            delete_summary = {
                "dry_run": dry_run,
                "deleted_archive_count": 0,
                "delete_skipped_count": 0,
                "skipped": True,
            }
        logger.log(delete_summary, title="delete archive summary")
        emit(log_level, "INFO", f"[delete] deleted={delete_summary['deleted_archive_count']}")

        if remove_empty_folder:
            empty_summary = remove_empty_dirs(dataset_dir, dry_run=dry_run)
        else:
            empty_summary = {"removed_empty_dir_count": 0, "dry_run": dry_run, "skipped": True}
        logger.log(empty_summary, title="remove empty dirs summary")

        return {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "zip_list_file": str(zip_list_file),
            "delete_archive": delete_archive_flag,
            "remove_empty_folder": remove_empty_folder,
            "dry_run": dry_run,
            "log_level": log_level,
            "delete_archive_summary": delete_summary,
            "remove_empty_dirs_summary": empty_summary,
        }
