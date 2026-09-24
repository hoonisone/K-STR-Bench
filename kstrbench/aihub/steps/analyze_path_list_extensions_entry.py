"""
경로 리스트(txt)를 읽어 확장자별 개수와 총 수량을 집계해 텍스트로 저장.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..helpers.step_logger import StepLogger, emit, normalize_log_level


def run() -> None:
    run_from_config({})


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


def _extract_ext(line: str, case_sensitive: bool) -> str:
    path_text = line.strip().replace("\\", "/")
    if not path_text:
        return ""
    name = path_text.rsplit("/", 1)[-1]
    if "." not in name:
        return "(no_ext)"
    ext = name.rsplit(".", 1)[-1].strip()
    if not ext:
        return "(no_ext)"
    if not case_sensitive:
        ext = ext.lower()
    return f".{ext}"


def _display_path(path_obj: Path, base_dir: Path | None, store_relative_paths: bool) -> str:
    if not store_relative_paths or base_dir is None:
        return str(path_obj)
    try:
        return path_obj.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path_obj)


class AnalyzePathListExtensionsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    logger = StepLogger("analyze_path_list_extensions_entry")

    list_file_raw = params.get("list_file")
    out_file_raw = params.get("out_file")
    case_sensitive = _coerce_bool(params.get("case_sensitive", True), default=True)
    store_relative_paths = _coerce_bool(
        params.get("store_relative_paths", False),
        default=False,
    )
    path_base_dir_raw = params.get("path_base_dir")
    if not list_file_raw:
        raise ValueError("params.list_file is required")
    if not out_file_raw:
        raise ValueError("params.out_file is required")
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    list_file = Path(str(list_file_raw)).expanduser().resolve()
    out_file = Path(str(out_file_raw)).expanduser().resolve()
    path_base_dir = (
        Path(str(path_base_dir_raw)).expanduser().resolve() if path_base_dir_raw else None
    )
    if not list_file.is_file():
        raise FileNotFoundError(f"list_file not found: {list_file}")

    with list_file.open("r", encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]

    non_empty_rows = [r for r in rows if r.strip()]
    counter: Counter[str] = Counter(_extract_ext(r, case_sensitive) for r in non_empty_rows)
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    out_lines: list[str] = []
    out_lines.append(
        f"input_file: {_display_path(list_file, path_base_dir, store_relative_paths)}"
    )
    out_lines.append(f"case_sensitive: {case_sensitive}")
    out_lines.append(f"total_lines: {len(rows)}")
    out_lines.append(f"non_empty_lines: {len(non_empty_rows)}")
    out_lines.append("")
    out_lines.append("[extension_counts]")
    for ext, cnt in sorted_items:
        out_lines.append(f"{ext}: {cnt}")
    out_lines.append("")
    out_lines.append(f"unique_extension_count: {len(sorted_items)}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_lines) + "\n")

    extension_counts = {ext: cnt for ext, cnt in sorted_items}
    logger.log(
        {
            "input_file": str(list_file),
            "out_file": str(out_file),
            "case_sensitive": case_sensitive,
            "total_lines": len(rows),
            "non_empty_lines": len(non_empty_rows),
            "unique_extension_count": len(sorted_items),
            "extension_counts": extension_counts,
        },
        title="analyze path list extensions counts",
    )

    result = {
        "ok": True,
        "list_file": str(list_file),
        "out_file": str(out_file),
        "case_sensitive": case_sensitive,
        "store_relative_paths": store_relative_paths,
        "path_base_dir": str(path_base_dir) if path_base_dir else None,
        "total_lines": len(rows),
        "non_empty_lines": len(non_empty_rows),
        "unique_extension_count": len(sorted_items),
        "log_level": log_level,
    }
    logger.log(result, title="analyze path list extensions result")
    emit(log_level, "INFO", f"[analyze_ext] input={list_file}")
    emit(log_level, "INFO", f"[analyze_ext] out={out_file}")
    emit(log_level, "INFO", f"[analyze_ext] total={len(rows)}, non_empty={len(non_empty_rows)}")
    emit(log_level, "INFO", f"[analyze_ext] unique_extension_count={len(sorted_items)}")
    return result

