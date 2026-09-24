from __future__ import annotations

from .extract_empty_dirs_from_list_entry import ExtractEmptyDirsFromListStep


class FilterEmptyOrMissingDirsFromListStep(ExtractEmptyDirsFromListStep):
    """빈 디렉터리/누락 경로 필터링용 명시적 이름 래퍼."""

