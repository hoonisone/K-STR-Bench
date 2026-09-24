# Dataset structure

K-STR-Bench stores every subset as two units: **scene** (the full image) and **text** (each text instance inside it). **AIHub_Standardized**, **GIST_Signboard**, and **KAIST_Corrected** all follow this layout.

```text
{subset}/
├── scene_images/
├── text_images/
├── scene.csv
└── text.v*.csv
```

## IDs

Each scene and each text has a stable ID. The ID starts with the subset name (`aihub_…`, `gist_…`, `kaist_…`), so values stay unique if the three tables are concatenated.

A text row also stores `scene_id`, so a text instance always points back to its source image.

## CSV is the source of truth

Annotations, splits, and extra metadata live in CSV, not in the training list files.

| Table | One row is | Key |
| --- | --- | --- |
| `scene.csv` | one full image | `scene_id` |
| `text.v*.csv` | one text instance | `text_id` |

When you need image-level fields (path, size, capture metadata) together with a transcription, join the two tables on `scene_id`. Versioned text tables (`text.v1.0.csv`, `text.v2.0.csv`) keep label revisions without renaming images.

## TXT is the training export

For STR training and evaluation, `kstrbench.make_label` reads the text CSV and writes tab-separated lists:

```text
{subset}/text_images/…/{text_id}.jpg<TAB>{label}
```

Those files go under `{subset}/labels/`: `train.txt`, `val.txt`, `test.txt`, and one file per challenge tag. Use them as `labelfile`. Do not edit the `.txt` lists by hand; change the CSV, then export again.
