---
pretty_name: K-STR-Bench
license: other
language:
  - ko
task_categories:
  - image-to-text
tags:
  - scene-text-recognition
  - korean
  - ocr
size_categories:
  - 1M<n<10M
---

# K-STR-Bench

K-STR-Bench is a unified, challenge-aware benchmark suite for Korean scene text recognition (STR).

It combines three complementary components: **AIHub-Standardized**, a reproducible large-scale STR resource with nearly 900K valid Korean text instances; **KAIST-Corrected**, a reconstructed and corrected legacy Korean scene-text dataset; and **GIST-Signboard**, a newly collected real-world signboard dataset covering challenging outdoor conditions.

The three components are organized under a common Korean-only cropped-STR protocol, supporting reproducible training, cross-dataset evaluation, and challenge-wise robustness analysis.

Code and instructions are available in the [GitHub repository](https://github.com/hoonisone/K-STR-Bench).

Myounghun Han, Jin-Hyuk Hong. ACCV 2026.

## Subsets

| Subset | Role | Scenes | Text | Korean-only | Train | Val | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIHub_Standardized | Train, val, and in-domain test | 533,469 | 1,937,042 | 897,055 | 717,669 | 89,716 | 89,670 |
| KAIST_Corrected | Evaluation only | 2,484 | 7,516 | 4,008 | — | — | 4,008 |
| GIST_Signboard | Evaluation only | 5,185 | 34,687 | 16,078 | — | — | 16,078 |
| Total | — | 541,138 | 1,979,245 | 917,141 | 717,669 | 89,716 | 109,756 |

Korean-only counts are the samples used for K-STR-Bench. Train, val, and test for AIHub_Standardized are that Korean-only split. KAIST_Corrected and GIST_Signboard are test only.

GIST_Signboard and KAIST_Corrected also carry challenge tags on the Korean-only test samples. A sample may have more than one tag.

| Subset | Artistic | Curve | Sailent | Incomplete | Multi-oriented | Low-visibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KAIST_Corrected | 563 | 128 | 446 | 56 | 411 | 282 |
| GIST_Signboard | 2,511 | 351 | 2,795 | 711 | 4,189 | 1,070 |

## What this repository contains

GIST_Signboard and KAIST_Corrected images and label tables are included.

AIHub images and the original AIHub annotation files are not included. This repository has the K-STR-Bench metadata, splits, and reviewed transcriptions. Rebuild AIHub_Standardized locally from the AIHub dataset **야외 실제 촬영 한글 이미지**.

Use these label files:

| Subset | File |
| --- | --- |
| AIHub_Standardized | `text.v2.0.csv` |
| KAIST_Corrected | `text.v2.0.csv` |
| GIST_Signboard | `text.v1.0.csv` |

Each text sample is keyed by `text_id`. A row points back to its scene with `scene_id`.

## Evaluation

Score matched predictions with word accuracy and jaso-level 1-NED. On GIST_Signboard and KAIST_Corrected, report the same two scores for each challenge tag.

The scoring command and file format are documented in the GitHub repository.

## License

This benchmark does not use one license for every file.

| Resource | Terms |
| --- | --- |
| GIST_Signboard images and annotations | CC BY 4.0 |
| K-STR-Bench metadata, ids, splits, and author-written labels | CC BY 4.0 |
| KAIST_Corrected | CC BY-SA 3.0 |
| Original AIHub images and annotations | Not redistributed. AIHub Terms of Use |

CC BY 4.0 covers only material created and released by the K-STR-Bench authors. It does not cover the original AIHub dataset.

## Citation

```bibtex
@inproceedings{han2026kstrbench,
    title     = {K-STR-Bench: A Challenge-Aware Benchmark Suite for Korean Scene Text Recognition},
    author    = {Han, Myounghun and Hong, Jin-Hyuk},
    booktitle = {Asian Conference on Computer Vision (ACCV)},
    year      = {2026}
}
```

Also cite the original AIHub and KAIST datasets when using those subsets.
