# AIHub Standardization

This folder turns the AIHub dataset **야외 실제 촬영 한글 이미지** into the K-STR-Bench `AIHub_Standardized` subset: cropped text images, scene images, and a patched `text.v2.0.csv`.

Download, disk space, and license notes are in the [root README](../README.md#2-prepare-aihub_standardized).

## What it produces

```text
<DATASET_DIR>/AIHub_Standardized/
├── scene_images/
├── text_images/
├── scene.csv
├── text.v1.0.csv
└── text.v2.0.csv
```

## Version history

| Version | What it is | Detail |
| --- | --- | --- |
| 1.0 | One row per text sample after cleaning the original AIHub data, with review columns added. File: `text.v1.0.csv` (1,936,540 rows). | Duplicate `text_id`s are removed: 404 ids, 502 extra rows deleted, first row kept. Bbox format, bbox value, language, and crop success are tagged only; those rows stay. `is_valid_bbox_format` is false for 63 rows, `is_valid_bbox_value` is false for 1,253 rows, and `is_valid_crop` is false for 10 rows. A crop image is written only when the label is Korean and both bbox flags are true. |
| **2.0** (current) | Reviewed transcriptions and train/val/test splits applied by `text.patch.v2.0.csv`. File: `text.v2.0.csv`. Use this version for K-STR-Bench. | The patch sets split on 897,055 text ids (train 717,669, val 89,716, test 89,670) and changes 660 labels. It does not repeat the 1.0 tags. Rows absent from the patch keep the 1.0 label and an empty split. |


## How it works

`standardize.py` runs these stages:

```text
unzip
→ validate_and_clean_dir
→ validate_and_clean_file
→ extract csv
→ process_text          # bbox check, crop, language filter
→ organize
→ clean_temp_dir
→ patch                 # text.v1.0.csv → text.v2.0.csv
```

| Path | Role |
| --- | --- |
| `AIHub_zip/` | Original AIHub download (kept) |
| `AIHub_unzip/` | Unpacked archives (temp) |
| `AIHub_dir_clean/` | Directory remap (temp) |
| `AIHub_dir_file_clean/` | File cleanup and crops (temp) |
| `AIHub_Standardized/` | Final STR subset |

The split stage is not enabled here. Train/val/test lists are written later by `kstrbench.make_label` into `AIHub_Standardized/labels/`.

## Run

From the repository root:

```bash
python -m kstrbench.aihub.standardize
```
