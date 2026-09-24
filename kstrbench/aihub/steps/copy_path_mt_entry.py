"""
파일/폴더 복사 엔트리 모듈.

kstrbench.aihub.helpers.copy_folder_mt.run_from_config를 파이프라인 스텝에서 호출할 수 있게 연결.
"""

from __future__ import annotations

from typing import Any

from ..helpers.copy_folder_mt import run_from_config as _run_copy


class CopyPathMTStep:
    """class/init_args 스키마를 위한 래퍼 스텝."""

    def __init__(self, **init_args: Any) -> None:
        self._init_args = dict(init_args)

    def run(self) -> dict[str, Any]:
        return run_from_config(self._init_args)


def run() -> None:
    run_from_config({})


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    mapped = dict(params)

    # backward-compatible aliases
    if "src" not in mapped and "src_file" in mapped:
        mapped["src"] = mapped["src_file"]
    if "dst" not in mapped and "dst_file" in mapped:
        mapped["dst"] = mapped["dst_file"]

    return _run_copy(mapped)

