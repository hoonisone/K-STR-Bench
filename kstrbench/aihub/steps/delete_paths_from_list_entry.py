from __future__ import annotations

from typing import Any

from .delete_files import run_from_config


class DeletePathsFromListStep:
    """delete_list_file 기반 경로(파일/폴더) 삭제 엔트리."""

    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)

