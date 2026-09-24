from __future__ import annotations

from typing import Any

from .split_annotations_tag_balanced_group_entry2 import run_from_config as _run_v2


class SplitAnnotationsTagBalancedGroupStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return _run_v2(self.params)


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _run_v2(params)


def run() -> None:
    run_from_config({})
