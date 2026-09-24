from __future__ import annotations

import os
from pathlib import Path

AIHUB = "AIHub_Standardized"
GIST = "GIST_Signboard"
KAIST = "KAIST_Corrected"

TEXT_CSV = {
    AIHUB: "text.v2.0.csv",
    GIST: "text.v1.0.csv",
    KAIST: "text.v2.0.csv",
}

CHALLENGE_FILES = (
    ("KSTR_is_artistic", "artistic.txt"),
    ("KSTR_is_curve", "curve.txt"),
    ("KSTR_is_sailent", "sailent.txt"),
    ("KSTR_is_incomplete", "incomplete.txt"),
    ("KSTR_is_multi_oriented", "multi_oriented.txt"),
    ("KSTR_is_low_visibility", "low_visibility.txt"),
)


def dataset_root(explicit: str | None = None) -> Path:
    """Use an explicit path when one was passed in. Otherwise read DATASET_DIR."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("DATASET_DIR")
    if not env:
        raise RuntimeError(
            "Set the DATASET_DIR environment variable or pass a dataset path."
        )
    return Path(env)


def subset_dir(root: Path, subset: str) -> Path:
    return root / subset


def text_csv(root: Path, subset: str) -> Path:
    return subset_dir(root, subset) / TEXT_CSV[subset]


def labels_dir(root: Path, subset: str) -> Path:
    return subset_dir(root, subset) / "labels"


def split_txt(root: Path, subset: str, split: str) -> Path:
    return labels_dir(root, subset) / f"{split}.txt"


def challenge_txt(root: Path, subset: str, name: str) -> Path:
    return labels_dir(root, subset) / name


def aihub_zip(root: Path) -> Path:
    return root / "AIHub_zip"


def aihub_unzip(root: Path) -> Path:
    return root / "AIHub_unzip"


def aihub_dir_clean(root: Path) -> Path:
    return root / "AIHub_dir_clean"


def aihub_dir_file_clean(root: Path) -> Path:
    return root / "AIHub_dir_file_clean"


def aihub_standardized(root: Path) -> Path:
    return subset_dir(root, AIHUB)
