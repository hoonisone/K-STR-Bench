"""
AIHub scene / text column names.

Old AIHub tables map as:
  images.csv      -> scene.csv
  annotations.csv -> text.csv

`origin_image_path` belongs on scene only.
"""

from __future__ import annotations

from enum import Enum


class SceneCol(str, Enum):
    SCENE_ID = "scene_id"
    ORIGIN_DATASET = "origin_dataset"
    IMAGE_PATH = "image_path"
    ORIGIN_IMAGE_PATH = "origin_image_path"
    WIDTH = "width"
    HEIGHT = "height"
    SPLIT = "split"
    DATA_CAPTURED_DATE = "data_captured_date"


class TextCol(str, Enum):
    TEXT_ID = "text_id"
    SCENE_ID = "scene_id"
    TEXT_INDEX = "text_index"
    ORIGIN_DATASET = "origin_dataset"
    LABEL = "label"
    POINTS = "points"
    IMAGE_PATH = "image_path"
    LANGUAGE_CATEGORY = "language_category"
    LENGTH = "length"
    ASPECT_RATIO = "aspect_ratio"
    LINEAR_ASPECT_RATIO_BIN = "linear_aspect_ratio_bin"
    IS_VALID_BBOX_FORMAT = "is_valid_bbox_format"
    IS_VALID_BBOX_VALUE = "is_valid_bbox_value"
    IS_VALID_CROP = "is_valid_crop"
    SPLIT = "split"


ORIGIN_DATASET_AIHUB = "aihub"
SCENE_ID_PREFIX_AIHUB = "aihub"
