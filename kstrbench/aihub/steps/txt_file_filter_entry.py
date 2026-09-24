from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class TXTFileFilter:
    def __init__(
        self,
        source_file: Any | None = None,
        out_file: Any | None = None,
        cache_file: Any | None = None,
        keyword: Any | None = None,
        drop_filtered_file_from_source_list: Any = False,
        updated_source_file: Any | None = None,
        log_level: Any | None = None,
    ) -> None:
        self.source_file_raw = source_file
        self.out_file_raw = out_file
        self.cache_file_raw = cache_file
        self.keyword_raw = keyword
        self.drop_filtered_file_from_source_list_raw = drop_filtered_file_from_source_list
        self.updated_source_file_raw = updated_source_file
        self.log_level_raw = log_level

    def run(self) -> dict[str, Any]:
        if not self.source_file_raw:
            raise ValueError("source_file is required")
        if not self.out_file_raw:
            raise ValueError("out_file is required")
        if self.keyword_raw is None:
            raise ValueError("keyword is required")

        source_file = Path(str(self.source_file_raw)).expanduser().resolve()
        out_file = Path(str(self.out_file_raw)).expanduser().resolve()
        cache_file = (
            Path(str(self.cache_file_raw)).expanduser().resolve()
            if self.cache_file_raw
            else None
        )
        keyword = str(self.keyword_raw)
        if not source_file.is_file():
            raise FileNotFoundError(f"source_file not found: {source_file}")
        drop_filtered_from_source = coerce_bool(
            self.drop_filtered_file_from_source_list_raw, default=False
        )
        updated_source_file = (
            Path(str(self.updated_source_file_raw)).expanduser().resolve()
            if self.updated_source_file_raw
            else None
        )
        if drop_filtered_from_source and updated_source_file is None:
            raise ValueError(
                "updated_source_file is required when drop_filtered_file_from_source_list=true"
            )

        log_level = normalize_log_level(self.log_level_raw, default="INFO")

        total_count = 0
        matched_count = 0
        cache_hit = bool(cache_file and cache_file.is_file())
        if cache_hit:
            with cache_file.open("r", encoding="utf-8") as cached:
                matched_count = sum(1 for _ in cached)
            with source_file.open("r", encoding="utf-8") as src:
                total_count = sum(1 for _ in src)
            emit(log_level, "INFO", f"[txt_filter] cache hit: {cache_file}")
        else:
            lines: list[str] = []
            with source_file.open("r", encoding="utf-8") as src:
                for raw_line in src:
                    total_count += 1
                    if keyword in raw_line:
                        matched_count += 1
                        lines.append(raw_line.rstrip("\n"))

            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with cache_file.open("w", encoding="utf-8", newline="\n") as cached:
                    for line in lines:
                        cached.write(line + "\n")
                emit(log_level, "INFO", f"[txt_filter] cache saved: {cache_file}")

            if not cache_file:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with out_file.open("w", encoding="utf-8", newline="\n") as dst:
                    for line in lines:
                        dst.write(line + "\n")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file and cache_file.is_file() and cache_file != out_file:
            shutil.copyfile(cache_file, out_file)
        elif cache_file and cache_file == out_file and not cache_hit:
            # already written via cache path
            pass

        emit(log_level, "INFO", f"[txt_filter] source_file={source_file}")
        emit(log_level, "INFO", f"[txt_filter] out_file={out_file}")
        emit(log_level, "INFO", f"[txt_filter] cache_file={cache_file}")
        emit(log_level, "INFO", f"[txt_filter] keyword={keyword}")
        emit(log_level, "INFO", f"[txt_filter] matched={matched_count}/{total_count}")

        updated_source_count: int | None = None
        if drop_filtered_from_source and updated_source_file is not None:
            kept_lines: list[str] = []
            with source_file.open("r", encoding="utf-8") as src:
                for raw_line in src:
                    if keyword not in raw_line:
                        kept_lines.append(raw_line.rstrip("\n"))
            updated_source_file.parent.mkdir(parents=True, exist_ok=True)
            with updated_source_file.open("w", encoding="utf-8", newline="\n") as dst:
                for line in kept_lines:
                    dst.write(line + "\n")
            updated_source_count = len(kept_lines)
            emit(log_level, "INFO", f"[txt_filter] updated_source_file={updated_source_file}")
            emit(
                log_level,
                "INFO",
                f"[txt_filter] updated_source_count={updated_source_count}",
            )

        logger = StepLogger("txt_file_filter_entry")
        logger.log(
            {
                "source_file": str(source_file),
                "out_file": str(out_file),
                "cache_file": str(cache_file) if cache_file else None,
                "cache_hit": cache_hit,
                "keyword": keyword,
                "drop_filtered_file_from_source_list": drop_filtered_from_source,
                "updated_source_file": (
                    str(updated_source_file) if updated_source_file else None
                ),
                "updated_source_count": updated_source_count,
                "line_count": total_count,
                "matched_count": matched_count,
                "log_level": log_level,
            },
            title="txt filter result",
        )
        return {
            "ok": True,
            "source_file": str(source_file),
            "out_file": str(out_file),
            "cache_file": str(cache_file) if cache_file else None,
            "cache_hit": cache_hit,
            "keyword": keyword,
            "drop_filtered_file_from_source_list": drop_filtered_from_source,
            "updated_source_file": (
                str(updated_source_file) if updated_source_file else None
            ),
            "updated_source_count": updated_source_count,
            "line_count": total_count,
            "matched_count": matched_count,
            "log_level": log_level,
        }
