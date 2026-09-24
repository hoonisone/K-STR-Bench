from __future__ import annotations

from kstrbench.dataset_dir import GIST, KAIST, dataset_root, subset_dir, text_csv
from kstrbench.step import Pipeline
from kstrbench.steps.fill_csv_column_from_image_filenames import FillCsvColumnFromImageFilenamesStep
from kstrbench.steps.move_images_by_csv_condition import MoveImagesByCsvConditionStep

_ROOT = dataset_root()
KAIST_DATASET = str(subset_dir(_ROOT, KAIST))
KAIST_TEXT_CSV = str(text_csv(_ROOT, KAIST))
KAIST_IMAGE_DIR = KAIST_DATASET

GIST_DATASET = str(subset_dir(_ROOT, GIST))
GIST_TEXT_CSV = str(text_csv(_ROOT, GIST))
GIST_IMAGE_DIR = GIST_DATASET


CSV_ENCODING = "utf-8-sig"

challenge_columns = [
    "KSTR_is_artistic",
    # "KSTR_is_curve",
    # "KSTR_is_sailent",
    # "KSTR_is_incomplete",
    # "KSTR_is_multi_oriented",
    # "KSTR_is_low_visibility",
]

pipelines = {}
move_by_shape_size_steps = [
    MoveImagesByCsvConditionStep(
        name=f"move",
        csv_path=GIST_TEXT_CSV,
        image_path_col="image_path",
        data_dir=GIST_IMAGE_DIR,
        operation="copy",
        column_value_filters={
            "is_only_korean": True,
            "is_illegible": False,
            "KSTR_is_artistic": True,
        },
        exclude_column_value_filters={},
        new_data_dir=str(_ROOT / "tagging2" / "artistic"),
        csv_encoding=CSV_ENCODING,
    ),
    MoveImagesByCsvConditionStep(
        name=f"move",
        csv_path=KAIST_TEXT_CSV,
        image_path_col="image_path",
        data_dir=KAIST_IMAGE_DIR,
        operation="copy",
        column_value_filters={
            "is_only_korean": True,
            "is_illegible": False,
            "KSTR_is_artistic": True,
        },
        exclude_column_value_filters={},
        new_data_dir=str(_ROOT / "tagging2" / "artistic"),
        csv_encoding=CSV_ENCODING,
    )
]

pipelines["move"] = Pipeline(
    name="move",
    steps=move_by_shape_size_steps,
)



fill_from_image_dir_steps = []
for tag in ["both", "is_noisy_background", "is_stylized"]:
    fill_from_image_dir_steps.append(
        FillCsvColumnFromImageFilenamesStep(
            name="fill",
            csv_path=GIST_DATASET + r"\text.csv",
            output_csv_path=GIST_DATASET + r"\text.csv",
            image_dir=str(_ROOT / "tagging2" / tag),
            text_id_col="text_id",
            output_col=tag,
            output_value=True,
            csv_encoding=CSV_ENCODING,
        )
    )

pipelines["fill"] = Pipeline(
    name="fill",
    steps=fill_from_image_dir_steps,
)


if __name__ == "__main__":
    pipelines["move"].run()
    # pipelines["fill"].run()
