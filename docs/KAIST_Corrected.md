# KAIST_Corrected

**KAIST_Corrected** is a reconstructed word-level STR subset of the [KAIST Scene Text Database](http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database). It is an evaluation-only real-scene resource: smaller than AIHub_Standardized, and less hard than GIST_Signboard, but useful as an independent outdoor/indoor distribution.

Download, splits, and license notes are in the [root README](../README.md).

## What it is

KAIST is a legacy Korean–English scene text dataset collected from natural scenes in Korean streets and shops. Images were captured indoors and outdoors under varied lighting (day, night, strong artificial light) with a high-resolution digital camera or a low-resolution mobile phone, then resized to 640×480. The original release provides character-level boxes and transcriptions, with word-level grouping, plus pixel-level segmentation masks. It was not released as a word-level cropped STR benchmark.

K-STR-Bench reconstructs that resource for cropped STR evaluation:

- Merge character-level annotations into word-level transcriptions.
- Convert boxes into word-level **text** crops from each **scene** image.
- Correct reading-order errors, missing or duplicated characters, wrong transcriptions, and inaccurate boxes.
- Add readable text that is visible in the image but missing from the original labels.

The original copy used here has 2,484 **scene** images and 5,499 text instances (3,069 Korean). After correction, 2,367 existing labels were fixed and missing instances were added, yielding 7,516 **text** samples, of which 4,008 Korean-only samples are used in K-STR-Bench.

Challenge tags and wildcard transcriptions used by K-STR-Bench are stored in `text.v2.0.csv` (`KSTR_is_*`).

## Statistics


|                                     | Count |
| ----------------------------------- | ----- |
| scene                               | 2,484 |
| text                                | 7,516 |
| Korean-only text (K-STR-Bench eval) | 4,008 |


KAIST_Corrected is evaluation-only. `kstrbench.make_label` writes `KAIST_Corrected/labels/test.txt` and one file per challenge tag in the same folder.

## Layout

```text
KAIST_Corrected/
├── scene_images/
├── text_images/
├── scene.csv
└── text.v2.0.csv
```



## Version history


| Version           | What it is                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| 1.0               | Original KAIST labels converted to the K-STR-Bench table format. File: `text.v1.0.csv`.                     |
| **2.0** (current) | Reviewed labels, with missing readable text added. File: `text.v2.0.csv`. Use this version for K-STR-Bench. |




## License

KAIST-derived images and annotations remain under **CC BY-SA 3.0**. Organization, preprocessing, and labels in this subset may differ from the original release. See the [root README](../README.md#license).

## Citation

Cite the original KAIST dataset and the K-STR-Bench paper. The K-STR-Bench BibTeX entry is in the [root README](../README.md#citation).

```bibtex
@article{jung2011touchtt,
    title   = {Touch {TT}: Scene Text Extractor Using Touchscreen Interface},
    author  = {Jung, Jehyun and Lee, SeongHun and Cho, Min Su and Kim, Jin Hyung},
    journal = {ETRI Journal},
    volume  = {33},
    number  = {1},
    pages   = {78--88},
    year    = {2011}
}
```

