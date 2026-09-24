from __future__ import annotations

import argparse
from pathlib import Path

from kstrbench.dataset_dir import (
    aihub_dir_clean,
    aihub_dir_file_clean,
    aihub_standardized,
    aihub_unzip,
    aihub_zip,
    dataset_root,
)
from kstrbench.step import Pipeline

from .columns import SCENE_ID_PREFIX_AIHUB
from .steps.add_arithmetic_column_entry import AddArithmeticColumnStep
from .steps.add_text_language_category_column_entry import AddTextLanguageCategoryColumnStep
from .steps.add_text_length_column_entry import AddTextLengthColumnStep
from .steps.analyze_dir_list_status_entry import AnalyzeDirListStatusStep
from .steps.analyze_path_list_extensions_entry import AnalyzePathListExtensionsStep
from .steps.apply_path_mapping_csv_entry import ApplyPathMappingCsvStep
from .steps.apply_text_update_csv import ApplyTextUpdateCsvStep
from .steps.assert_path_exist import AssertPathExist
from .steps.assign_annotation_image_paths_entry import AssignAnnotationImagePathsStep
from .steps.check_mapped_path_list_match_entry import CheckMappedPathListMatchStep
from .steps.copy_path_mt_entry import CopyPathMTStep
from .steps.crop_text_images_from_annotations_entry import CropTextImagesFromAnnotationsStep
from .steps.delete_path import DeletePathStep
from .steps.drop_duplicate_text_ids_entry import DropDuplicateTextIdsStep
from .steps.delete_paths_from_list_entry import DeletePathsFromListStep
from .steps.exclude_annotations_by_id_list_entry import ExcludeAnnotationsByIdListStep
from .steps.extract_bbox_size_columns_entry import ExtractBboxSizeColumnsStep
from .steps.export_image_label_pair_metadata_csv_entry import (
    ExportImageLabelPairMetadataCsvStep,
)
from .steps.filter_empty_or_missing_dirs_from_list_entry import (
    FilterEmptyOrMissingDirsFromListStep,
)
from .steps.legacy_step import wrap
from .steps.make_global_id_rename_map_from_match_pair_entry import (
    MakeGlobalIdRenameMapFromMatchPairStep,
)
from .steps.make_image_path_prefix_column_entry import MakeImagePathPrefixColumnStep
from .steps.make_leaf_dir_list_entry import MakeLeafDirListStep
from .steps.make_linear_aspect_ratio_bin_entry import MakeLinearAspectRatioBinStep
from .steps.merge_image_columns_to_annotations_entry import (
    MergeImageColumnsToAnnotationsStep,
)
from .steps.make_path_list import MakePathLIst
from .steps.map_aihub_tables_to_scene_text import MapAihubTablesToSceneTextStep
from .steps.remap_path_list_dirs_entry import RemapPathListDirsStep
from .steps.remap_path_list_extensions_entry import RemapPathListExtensionsStep
from .steps.replace_csv_path_prefix import ReplaceCsvPathPrefixStep
from .steps.select_csv_columns import SelectCsvColumnsStep
from .steps.split_annotations_tag_balanced_group_entry import (
    SplitAnnotationsTagBalancedGroupStep,
)
from .steps.subtract_path_list_entry import SubtractPathListStep
from .steps.txt_file_filter_entry import TXTFileFilter
from .steps.unzip_archives_entry import UnzipArchivesStep
from .steps.validate_annotation_bbox_entry import ValidateAnnotationBboxStep

_THIS_DIR = Path(__file__).resolve().parent


def _explicit_dataset_dir() -> str | None:
    parser = argparse.ArgumentParser(
        description="Standardize the AIHub subset for K-STR-Bench"
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dataset root folder. If omitted, use the DATASET_DIR environment variable.",
    )
    if __name__ != "__main__":
        return None
    return parser.parse_args().dataset_dir


_ROOT = dataset_root(_explicit_dataset_dir())
DATASET_DIR = str(_ROOT)
RESOURCES_DIR = str(_THIS_DIR / "resources")

LATEST_VERSION = "v2.0"

ARCHIVE_DIR = str(aihub_zip(_ROOT))
AIHUB_DIR = str(aihub_unzip(_ROOT))
DIR_CLEAN_DIR = str(aihub_dir_clean(_ROOT))
DIR_FILE_CLEAN_DIR = str(aihub_dir_file_clean(_ROOT))

SCENE_CSV = DIR_FILE_CLEAN_DIR + "/scene.csv"
TEXT_CSV = DIR_FILE_CLEAN_DIR + "/text.csv"
TEXT_IMAGE_DIR = DIR_FILE_CLEAN_DIR + "/text_images"

STANDARDIZE_DIR = str(aihub_standardized(_ROOT))
STANDARDIZE_SCENE_CSV = STANDARDIZE_DIR + "/scene.csv"
STANDARDIZE_TEXT_CSV_V1 = STANDARDIZE_DIR + "/text.v1.0.csv"
STANDARDIZE_TEXT_PATCH_CSV = STANDARDIZE_DIR + f"/text.patch.{LATEST_VERSION}.csv"
STANDARDIZE_TEXT_CSV_LATEST = STANDARDIZE_DIR + f"/text.{LATEST_VERSION}.csv"

VALID_TEXT_FILTERS = [
    {"column": "language_category", "op": "eq", "value": "KOR"},
    {"column": "is_valid_bbox_format", "op": "eq", "value": "True"},
    {"column": "is_valid_bbox_value", "op": "eq", "value": "True"},
]
VALID_CROP_FILTERS = VALID_TEXT_FILTERS + [
    {"column": "is_valid_crop", "op": "eq", "value": "True"},
]
SPLIT_MERGE_COLUMNS = [
    "dir_path_4",
    "area",
    "device",
    "illuminance",
    "light",
    "metaclass",
    "outline",
    "weather",
    "wordcolor",
    "wordfont",
    "wordorientation",
    "wordsize",
]
SPLIT_GROUP_COLUMNS = ["length", "linear_aspect_ratio_bin", *SPLIT_MERGE_COLUMNS]

DIR_MAPPING_FILE = RESOURCES_DIR + "/AIHub_v1.1_unzip___dir_mapping.csv"
LEAF_DIR_LIST_FILE = RESOURCES_DIR + "/AIHub_v1.0_archive___leaf_dir_list.txt"
ZIP_FILE_PATH_LIST_FILE = RESOURCES_DIR + "/AIHub_v1.0_archive___zip_file_path_list.txt"

pipelines: dict[str, Pipeline] = {}

pipelines["unzip"] = Pipeline(
    name="unzip",
    steps=[
        wrap(
            MakeLeafDirListStep,
            name="outdoor_make_leaf_dir_list",
            dataset_dir=ARCHIVE_DIR,
            out_list_file=ARCHIVE_DIR + "/metadata/leaf_dir_list.txt",
            cache_file=LEAF_DIR_LIST_FILE,
            store_relative=True,
            include_root_if_leaf=True,
            log_level="INFO",
        ),
        wrap(
            AssertPathExist,
            name="outdoor_assert_leaf_dirs_exist_temp",
            dataset_dir=ARCHIVE_DIR,
            list_file=ARCHIVE_DIR + "/metadata/leaf_dir_list.txt",
            error_message="Some paths listed in leaf_dir_list do not exist.",
            max_print_missing=10,
            log_level="INFO",
        ),
        wrap(
            MakePathLIst,
            name="make_all_path_list",
            dataset_dir=ARCHIVE_DIR,
            scan_roots_file=ARCHIVE_DIR + "/metadata/leaf_dir_list.txt",
            out_list_file=ARCHIVE_DIR + "/metadata/zip_file_path_list.txt",
            cache_file=ZIP_FILE_PATH_LIST_FILE,
            extensions=["zip"],
            num_process=15,
            num_thread=2,
            store_relative=True,
            show_progress=True,
            skip_if_exists=True,
            log_level="INFO",
        ),
        wrap(
            AssertPathExist,
            name="outdoor_assert_zip_paths_exist",
            dataset_dir=ARCHIVE_DIR,
            list_file=ARCHIVE_DIR + "/metadata/zip_file_path_list.txt",
            error_message="Some paths listed in zip_file_path_list do not exist.",
            max_print_missing=10,
            log_level="INFO",
        ),
        wrap(
            UnzipArchivesStep,
            name="outdoor_unzip_archives",
            out_dataset_dir=AIHUB_DIR,
            source_dataset_dir=ARCHIVE_DIR,
            zip_list_valid_file=ARCHIVE_DIR + "/metadata/zip_file_path_list.txt",
            dry_run=False,
            overwrite=False,
            num_process=6,
            num_thread=8,
            log_level="INFO",
        ),
    ],
)

pipelines["validate_and_clean_dir"] = Pipeline(
    name="validate_and_clean_dir",
    steps=[
        wrap(
            MakeLeafDirListStep,
            name="outdoor_make_leaf_dir_list",
            dataset_dir=AIHUB_DIR,
            out_list_file=AIHUB_DIR + "/metadata/leaf_dir_list.txt",
            cache_file=None,
            store_relative=True,
            include_root_if_leaf=True,
            log_level="INFO",
        ),
        wrap(
            AnalyzeDirListStatusStep,
            name="AIHub_STD___analyze_leaf_dir_status",
            dataset_dir=AIHUB_DIR,
            list_file=AIHUB_DIR + "/metadata/leaf_dir_list.txt",
            out_file=AIHUB_DIR + "/analysis/leaf_dir_list_status1.txt",
            show_progress=True,
            sample_limit=20,
            log_level="INFO",
        ),
        wrap(
            FilterEmptyOrMissingDirsFromListStep,
            name="AIHub_STD___filter_empty_or_missing_dirs",
            dataset_dir=AIHUB_DIR,
            source_list_file=AIHUB_DIR + "/metadata/leaf_dir_list.txt",
            out_empty_list_file=AIHUB_DIR + "/metadata/empty_leaf_dir_list.txt",
            drop_filtered_file_from_source_list=True,
            updated_source_file=AIHUB_DIR + "/metadata/leaf_dir_list.txt",
            include_missing_paths=True,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            DeletePathsFromListStep,
            name="AIHub_STD___delete_paths_from_list",
            dataset_dir=AIHUB_DIR,
            delete_list_file=AIHUB_DIR + "/metadata/empty_leaf_dir_list.txt",
            remove_empty_folder=True,
            dry_run=False,
            show_progress=True,
            num_process=1,
            num_thread=1,
            log_level="INFO",
        ),
        wrap(
            AnalyzeDirListStatusStep,
            name="AIHub_STD___analyze_leaf_dir_status",
            dataset_dir=AIHUB_DIR,
            list_file=AIHUB_DIR + "/metadata/leaf_dir_list.txt",
            out_file=AIHUB_DIR + "/analysis/leaf_dir_list_status2.txt",
            show_progress=True,
            sample_limit=20,
            log_level="INFO",
        ),
        wrap(
            CopyPathMTStep,
            name="AIHub_STD___copy_dir_mapping_to_dataset",
            src=DIR_MAPPING_FILE,
            dst=AIHUB_DIR + "/metadata/dir_mapping.csv",
            operation="copy",
            show_progress=True,
            overwrite=True,
        ),
        wrap(
            RemapPathListDirsStep,
            name="AIHub_STD___remap_path_list_dirs",
            dataset_dir=AIHUB_DIR,
            out_dataset_dir=DIR_CLEAN_DIR,
            dir_mapping_file=AIHUB_DIR + "/metadata/dir_mapping.csv",
            origin_col="origin_path",
            new_col="new_path",
            out_changed_list_file=DIR_CLEAN_DIR + "/metadata/leaf_dir_list.txt",
            path_type="dir",
            num_process=1,
            num_thread=1,
            skip_missing_files=True,
            dry_run=False,
            show_progress=True,
            log_level="INFO",
        ),
    ],
)

pipelines["validate_and_clean_file"] = Pipeline(
    name="validate_and_clean_file",
    steps=[
        wrap(
            MakeLeafDirListStep,
            name="outdoor_make_leaf_dir_list",
            dataset_dir=DIR_CLEAN_DIR,
            out_list_file=DIR_CLEAN_DIR + "/metadata/leaf_dir_list.txt",
            cache_file=None,
            skip_if_exists=True,
            store_relative=True,
            include_root_if_leaf=True,
            log_level="INFO",
        ),
        wrap(
            MakePathLIst,
            name="make_all_path_list",
            dataset_dir=DIR_CLEAN_DIR,
            scan_roots_file=DIR_CLEAN_DIR + "/metadata/leaf_dir_list.txt",
            out_list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            extensions=None,
            num_process=8,
            num_thread=6,
            store_relative=True,
            show_progress=True,
            skip_if_exists=True,
            log_level="INFO",
        ),
        wrap(
            AnalyzePathListExtensionsStep,
            name="AIHub_STD___analyze_all_path_extensions",
            list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_CLEAN_DIR + "/analysis/all_file_ext_stats.txt",
            case_sensitive=True,
            path_base_dir=DIR_CLEAN_DIR,
            store_relative_paths=True,
            log_level="INFO",
        ),
        wrap(
            TXTFileFilter,
            name="outdoor_filter_leaf_dir_list",
            source_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_CLEAN_DIR + "/metadata/zip_file_path_list.txt",
            cache_file=None,
            keyword=".zip",
            drop_filtered_file_from_source_list=True,
            updated_source_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            log_level="INFO",
        ),
        wrap(
            DeletePathsFromListStep,
            name="AIHub_STD___delete_paths_from_list",
            dataset_dir=DIR_CLEAN_DIR,
            delete_list_file=DIR_CLEAN_DIR + "/metadata/zip_file_path_list.txt",
            remove_empty_folder=True,
            dry_run=False,
            show_progress=True,
            num_process=1,
            num_thread=1,
            log_level="INFO",
        ),
        wrap(
            RemapPathListExtensionsStep,
            name="AIHub_STD___remap_path_list_extensions",
            dataset_dir=DIR_CLEAN_DIR,
            list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            ext_mapping={
                "JPG": "jpg",
                "JPEG": "jpg",
                "jpeg": "jpg",
            },
            num_process=8,
            num_thread=24,
            skip_missing_files=False,
            dry_run=False,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            TXTFileFilter,
            name="make_image_file_path_list",
            source_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            cache_file=None,
            keyword=".jpg",
            log_level="INFO",
        ),
        wrap(
            TXTFileFilter,
            name="make_label_file_path_list",
            source_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            cache_file=None,
            keyword=".json",
            log_level="INFO",
        ),
        wrap(
            CheckMappedPathListMatchStep,
            name="AIHub_STD___check_mapped_image_label_match",
            image_list_file=DIR_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            label_list_file=DIR_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            out_match_pair_file=None,
            out_unmatch_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            keyword_mapping={
                "images": "labels",
                ".jpg": ".json",
            },
            show_progress=True,
            sample_limit=50,
            log_level="INFO",
        ),
        wrap(
            DeletePathsFromListStep,
            name="AIHub_STD___delete_paths_from_list",
            dataset_dir=DIR_CLEAN_DIR,
            delete_list_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            remove_empty_folder=True,
            dry_run=False,
            show_progress=True,
            num_process=1,
            num_thread=1,
            log_level="INFO",
        ),
        wrap(
            SubtractPathListStep,
            name="AIHub_STD___subtract_unmatched_from_all_list",
            source_list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            remove_list_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            out_list_file=DIR_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            normalize_slashes=True,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            SubtractPathListStep,
            name="AIHub_STD___subtract_unmatched_from_image_list",
            source_list_file=DIR_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            remove_list_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            out_list_file=DIR_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            normalize_slashes=True,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            SubtractPathListStep,
            name="AIHub_STD___subtract_unmatched_from_label_list",
            source_list_file=DIR_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            remove_list_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            out_list_file=DIR_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            normalize_slashes=True,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            CheckMappedPathListMatchStep,
            name="AIHub_STD___check_mapped_image_label_match",
            image_list_file=DIR_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            label_list_file=DIR_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            out_unmatch_file=DIR_CLEAN_DIR + "/metadata/unmatch_file_list1.txt",
            out_match_pair_file=DIR_CLEAN_DIR + "/metadata/match_file_pair.csv",
            keyword_mapping={
                "images": "labels",
                ".jpg": ".json",
            },
            show_progress=True,
            sample_limit=50,
            log_level="INFO",
        ),
        wrap(
            MakeGlobalIdRenameMapFromMatchPairStep,
            name="AIHub_STD___make_global_id_rename_map_from_match_pair",
            match_pair_csv_file=DIR_CLEAN_DIR + "/metadata/match_file_pair.csv",
            out_mapping_csv_file=DIR_CLEAN_DIR + "/metadata/file_path_mapping.csv",
            image_col="image_file",
            label_col="label_file",
            origin_col="origin_path",
            new_col="new_path",
            start_id=1,
            id_padding=0,
            id_prefix=SCENE_ID_PREFIX_AIHUB,
            normalize_slashes=True,
            log_level="INFO",
        ),
        wrap(
            ApplyPathMappingCsvStep,
            name="AIHub_STD___apply_file_path_mapping_csv",
            dataset_dir=DIR_CLEAN_DIR,
            out_dataset_dir=DIR_FILE_CLEAN_DIR,
            mapping_csv_file=DIR_CLEAN_DIR + "/metadata/file_path_mapping.csv",
            origin_col="origin_path",
            new_col="new_path",
            out_changed_list_file=DIR_FILE_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            skip_missing_files=True,
            dry_run=False,
            show_progress=True,
            log_level="INFO",
            num_process=4,
            num_thread=2,
        ),
    ],
)

pipelines["exract_scv"] = Pipeline(
    name="exract_scv",
    steps=[
        wrap(
            MakeLeafDirListStep,
            name="make_leaf_dir_list",
            dataset_dir=DIR_FILE_CLEAN_DIR,
            out_list_file=DIR_FILE_CLEAN_DIR + "/metadata/leaf_dir_list.txt",
            cache_file=None,
            skip_if_exists=False,
            store_relative=True,
            include_root_if_leaf=True,
            log_level="INFO",
        ),
        wrap(
            MakePathLIst,
            name="make_all_path_list",
            dataset_dir=DIR_FILE_CLEAN_DIR,
            scan_roots_file=DIR_FILE_CLEAN_DIR + "/metadata/leaf_dir_list.txt",
            out_list_file=DIR_FILE_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            extensions=None,
            num_process=8,
            num_thread=6,
            store_relative=True,
            show_progress=True,
            skip_if_exists=True,
            log_level="INFO",
        ),
        wrap(
            TXTFileFilter,
            name="make_image_file_path_list",
            source_file=DIR_FILE_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_FILE_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            cache_file=None,
            keyword=".jpg",
            log_level="INFO",
        ),
        wrap(
            TXTFileFilter,
            name="make_label_file_path_list",
            source_file=DIR_FILE_CLEAN_DIR + "/metadata/all_file_path_list.txt",
            out_file=DIR_FILE_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            cache_file=None,
            keyword=".json",
            log_level="INFO",
        ),
        wrap(
            CheckMappedPathListMatchStep,
            name="AIHub_STD___check_mapped_image_label_match",
            image_list_file=DIR_FILE_CLEAN_DIR + "/metadata/image_file_path_list.txt",
            label_list_file=DIR_FILE_CLEAN_DIR + "/metadata/label_file_path_list.txt",
            out_unmatch_file=DIR_FILE_CLEAN_DIR + "/metadata/unmatch_file_list.txt",
            out_match_pair_file=DIR_FILE_CLEAN_DIR + "/metadata/match_file_pair.csv",
            keyword_mapping={
                "images": "labels",
                ".jpg": ".json",
            },
            show_progress=True,
            sample_limit=50,
            log_level="INFO",
        ),
        wrap(
            ExportImageLabelPairMetadataCsvStep,
            name="AIHub_STD___export_pair_metadata_and_annotations_csv",
            dataset_dir=DIR_FILE_CLEAN_DIR,
            match_pair_csv_file=DIR_FILE_CLEAN_DIR + "/metadata/match_file_pair.csv",
            image_col="image_file",
            label_col="label_file",
            annotation_image_id_col="image_id",
            annotation_idx_col="inner_ann_id",
            out_image_csv_file=DIR_FILE_CLEAN_DIR + "/images.csv",
            out_annotation_csv_file=DIR_FILE_CLEAN_DIR + "/annotations.csv",
            show_progress=True,
            skip_missing_files=False,
            sort_metadata_keys=True,
            num_process=8,
            num_thread=2,
            log_level="INFO",
        ),
        MapAihubTablesToSceneTextStep(
            name="AIHub_STD___map_tables_to_scene_text",
            in_image_csv_file=DIR_FILE_CLEAN_DIR + "/images.csv",
            in_annotation_csv_file=DIR_FILE_CLEAN_DIR + "/annotations.csv",
            out_scene_csv_file=SCENE_CSV,
            out_text_csv_file=TEXT_CSV,
        ),
    ],
)

pipelines["process_text"] = Pipeline(
    name="process_text",
    steps=[
        wrap(
            DropDuplicateTextIdsStep,
            name="drop_duplicate_text_ids",
            in_csv_file=TEXT_CSV,
            out_csv_file=TEXT_CSV,
            text_id_col="text_id",
            keep="first",
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            ValidateAnnotationBboxStep,
            name="validate_bbox",
            in_csv_file=TEXT_CSV,
            out_csv_file=TEXT_CSV,
            out_stats_csv_file=DIR_FILE_CLEAN_DIR + "/analysis/text_bbox_validation_stats.csv",
            bbox_col="points",
            drop_invalid_bbox_format=False,
            drop_invalid_bbox_value=False,
            check_valid_bbox_format=True,
            check_valid_bbox_value=True,
            is_valid_bbox_format_col="is_valid_bbox_format",
            is_valid_bbox_value_col="is_valid_bbox_value",
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            AssignAnnotationImagePathsStep,
            name="assign_crop_image_path",
            annotation_csv_file=TEXT_CSV,
            out_annotation_csv_file=TEXT_CSV,
            source_image_id_col="scene_id",
            annotation_id_col="text_id",
            out_image_path_col="image_path",
            text_image_root="text_images",
            bin_size=1000,
            text_img_ext="jpg",
            overwrite_existing=True,
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            AddTextLanguageCategoryColumnStep,
            name="add_language_category",
            in_csv_file=TEXT_CSV,
            out_csv_file=TEXT_CSV,
            label_column="label",
            out_column="language_category",
            show_progress=True,
            log_level="INFO",
        ),
        wrap(
            CropTextImagesFromAnnotationsStep,
            name="crop_text_images",
            dataset_dir=DIR_FILE_CLEAN_DIR,
            out_dataset_dir=DIR_FILE_CLEAN_DIR,
            image_csv_file=SCENE_CSV,
            annotation_csv_file=TEXT_CSV,
            out_annotation_csv_file=TEXT_CSV,
            source_image_id_col="scene_id",
            source_image_path_col="image_path",
            annotation_id_col="text_id",
            bbox_col="points",
            text_image_root="text_images",
            filters=VALID_TEXT_FILTERS,
            out_error_csv_file=DIR_FILE_CLEAN_DIR + "/metadata/crop_error_samples.csv",
            text_img_qual=95,
            skip_if_out_dataset_exists=True,
            dry_run=False,
            skip_missing_files=False,
            show_progress=True,
            num_process=16,
            num_thread=2,
            ignore_exif_orientation=False,
            log_level="INFO",
        ),
        wrap(
            ExcludeAnnotationsByIdListStep,
            name="mark_crop_errors",
            in_csv_file=TEXT_CSV,
            out_csv_file=TEXT_CSV,
            error_csv_file=DIR_FILE_CLEAN_DIR + "/metadata/crop_error_samples.csv",
            annotation_id_col="text_id",
            is_valid_crop_col="is_valid_crop",
            show_progress=True,
            log_level="INFO",
        ),
    ],
)

pipelines["organize_aihub_standardized"] = Pipeline(
    name="organize_aihub_standardized",
    steps=[
        wrap(
            CopyPathMTStep,
            name="move_scene_images",
            src=DIR_FILE_CLEAN_DIR + "/images",
            dst=STANDARDIZE_DIR + "/scene_images",
            operation="move",
            show_progress=True,
            overwrite=True,
            num_process=4,
            num_thread=4,
        ),
        wrap(
            CopyPathMTStep,
            name="copy_text_images",
            src=TEXT_IMAGE_DIR,
            dst=STANDARDIZE_DIR + "/text_images",
            operation="move",
            show_progress=True,
            overwrite=True,
            num_process=4,
            num_thread=4,
        ),
        wrap(
            CopyPathMTStep,
            name="copy_scene_csv",
            src=SCENE_CSV,
            dst=STANDARDIZE_SCENE_CSV,
            operation="move",
            show_progress=True,
            overwrite=True,
        ),
        ReplaceCsvPathPrefixStep(
            name="rename_scene_image_path_prefix",
            in_csv_file=STANDARDIZE_SCENE_CSV,
            out_csv_file=STANDARDIZE_SCENE_CSV,
            column="image_path",
            old_prefix="images",
            new_prefix="scene_images",
        ),
        wrap(
            CopyPathMTStep,
            name="copy_text_csv",
            src=TEXT_CSV,
            dst=STANDARDIZE_TEXT_CSV_V1,
            operation="move",
            show_progress=True,
            overwrite=True,
        ),
        
    ],
)

# pipelines["split"] = Pipeline(
#     name="split",
#     steps=[
#         wrap(
#             AddTextLengthColumnStep,
#             name="add_text_length",
#             in_csv_file=STANDARDIZE_TEXT_CSV,
#             out_csv_file=STANDARDIZE_TEXT_CSV,
#             text_column="label",
#             out_column="length",
#             ignore_space=False,
#             filters=VALID_CROP_FILTERS,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         wrap(
#             ExtractBboxSizeColumnsStep,
#             name="extract_width_height_from_points",
#             in_csv_file=STANDARDIZE_TEXT_CSV,
#             out_csv_file=STANDARDIZE_TEXT_CSV,
#             bbox_col="points",
#             width_col="width",
#             height_col="height",
#             filters=VALID_CROP_FILTERS,
#             overwrite_existing=True,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         wrap(
#             AddArithmeticColumnStep,
#             name="add_aspect_ratio",
#             in_csv_file=STANDARDIZE_TEXT_CSV,
#             out_csv_file=STANDARDIZE_TEXT_CSV,
#             x_column="width",
#             y_column="height",
#             out_column="aspect_ratio",
#             filters=VALID_CROP_FILTERS,
#             operation="div",
#             aspect_ratio_precision=6,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         wrap(
#             MakeLinearAspectRatioBinStep,
#             name="add_linear_aspect_ratio_bin",
#             in_csv_file=STANDARDIZE_TEXT_CSV,
#             out_csv_file=STANDARDIZE_TEXT_CSV,
#             aspect_ratio_col="aspect_ratio",
#             linear_aspect_ratio_bin_col="linear_aspect_ratio_bin",
#             filters=VALID_CROP_FILTERS,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         wrap(
#             MakeImagePathPrefixColumnStep,
#             name="add_dir_path_4",
#             in_csv_file=STANDARDIZE_SCENE_CSV,
#             out_csv_file=STANDARDIZE_SCENE_CSV,
#             path_column="image_path",
#             out_column="dir_path_4",
#             prefix_token_count=4,
#             overwrite_existing=True,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         ReplaceCsvPathPrefixStep(
#             name="normalize_dir_path_4_prefix",
#             in_csv_file=STANDARDIZE_SCENE_CSV,
#             out_csv_file=STANDARDIZE_SCENE_CSV,
#             column="dir_path_4",
#             old_prefix="scene_images",
#             new_prefix="images",
#         ),
#         wrap(
#             MergeImageColumnsToAnnotationsStep,
#             name="merge_scene_columns_to_text",
#             image_csv_file=STANDARDIZE_SCENE_CSV,
#             annotation_csv_file=STANDARDIZE_TEXT_CSV,
#             out_annotation_csv_file=STANDARDIZE_TEXT_CSV,
#             image_id_col="scene_id",
#             annotation_image_id_col="scene_id",
#             merge_columns=SPLIT_MERGE_COLUMNS,
#             overwrite_existing=True,
#             strict_columns=False,
#             show_progress=True,
#             log_level="INFO",
#         ),
#         wrap(
#             SplitAnnotationsTagBalancedGroupStep,
#             name="split_by_scene",
#             in_csv_file=STANDARDIZE_TEXT_CSV,
#             out_csv_file=STANDARDIZE_TEXT_CSV,
#             split_col_name="split",
#             seed=42,
#             split_group_id_col="scene_id",
#             train_val_test_ratio=[8, 1, 1],
#             include_keywords=[],
#             exclude_keywords=[],
#             filters=VALID_CROP_FILTERS,
#             keyword_columns=["label"],
#             group_column=SPLIT_GROUP_COLUMNS,
#             show_progress=True,
#             log_level="INFO",
#         ),
#     ],
# )



PROTECTED_DIRS = [ARCHIVE_DIR, STANDARDIZE_DIR]

pipelines["clean_temp_dir"] = Pipeline(
    name="clean_temp_dir",
    steps=[
        DeletePathStep(
            name="delete_unzip_dir",
            path=AIHUB_DIR,
            missing_ok=True,
            must_be_under=DATASET_DIR,
            protected_paths=PROTECTED_DIRS,
        ),
        DeletePathStep(
            name="delete_dir_clean",
            path=DIR_CLEAN_DIR,
            missing_ok=True,
            must_be_under=DATASET_DIR,
            protected_paths=PROTECTED_DIRS,
        ),
        DeletePathStep(
            name="delete_dir_file_clean",
            path=DIR_FILE_CLEAN_DIR,
            missing_ok=True,
            must_be_under=DATASET_DIR,
            protected_paths=PROTECTED_DIRS,
        ),
    ],
)

pipelines["patch"] = Pipeline(
    name="patch",
    steps=[
        SelectCsvColumnsStep(
            name="select_text_columns",
            in_csv_file=STANDARDIZE_TEXT_CSV_V1,
            out_csv_file=STANDARDIZE_TEXT_CSV_LATEST,
            columns=["text_id", "scene_id", "origin_dataset", "image_path", "label", "points"],
        ),
        ApplyTextUpdateCsvStep(
            name="apply_text_patch",
            in_text_csv_file=STANDARDIZE_TEXT_CSV_LATEST,
            update_csv_file=STANDARDIZE_TEXT_PATCH_CSV,
            out_text_csv_file=STANDARDIZE_TEXT_CSV_LATEST,
        ),
    ],
)

def main() -> None:
    print(f"DATASET_DIR={DATASET_DIR}")
    pipelines["unzip"].run()
    pipelines["validate_and_clean_dir"].run()
    pipelines["validate_and_clean_file"].run()
    pipelines["exract_scv"].run()
    pipelines["process_text"].run()
    pipelines["organize_aihub_standardized"].run()
    pipelines["clean_temp_dir"].run()
    pipelines["patch"].run()
    
    ## pipelines["split"].run()



if __name__ == "__main__":
    main()
