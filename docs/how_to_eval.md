# How to evaluate

This page scores **one inference file** against one K-STR-Bench subset.

You provide the predictions. The tool looks up the official transcription for each sample, writes a scored CSV next to your file, and prints two averages: word accuracy and jaso-level 1-NED.

You do not put ground-truth labels in the inference file. Labels come from the subset CSV already on disk.

## Before you start

Install the package and set `DATASET_DIR` as described in the [root README](../README.md#usage). The command reads that folder. It does not take a separate dataset-path argument.

The subset you want to score must already be there:

| `--dataset` | Label file the tool reads |
| --- | --- |
| `AIHub_Standardized` | `<DATASET_DIR>/AIHub_Standardized/text.v2.0.csv` |
| `GIST_Signboard` | `<DATASET_DIR>/GIST_Signboard/text.v1.0.csv` |
| `KAIST_Corrected` | `<DATASET_DIR>/KAIST_Corrected/text.v2.0.csv` |

`GIST_Signboard` and `KAIST_Corrected` are in the release. `AIHub_Standardized` is produced locally. See [Prepare AIHub_Standardized](../README.md#2-prepare-aihub_standardized).

Each row of those files is one text sample. The sample id is `text_id`, for example `gist_12_3`. The official transcription is `label`.

## 1. Make one inference file

Use `.csv`, `.txt`, or `.jsonl`. The tool only needs two things from each row: which sample it is, and what the model predicted.

A sample is identified by `text_id`. That id is the image file name without the extension:

```text
GIST_Signboard/text_images/0/gist_12_3.jpg   →   gist_12_3
```

Slashes and backslashes are both fine. Folders in front of the file name are ignored.

### CSV

A CSV is the simplest input if you can write one. It needs a header and these two columns. Extra columns are kept.

```text
text_id,infer
gist_12_3,치킨
gist_12_4,카페
```

`text_id` is the sample id itself, not a path.

### TXT

Use this when your trainer already writes one line per image. Separate the image path and the prediction with a tab. Empty lines are skipped.

```text
GIST_Signboard/text_images/0/gist_12_3.jpg	치킨
GIST_Signboard/text_images/0/gist_12_4.jpg	카페
```

The tool turns this into a CSV before scoring. From `preds.txt` it writes `preds.csv`:

```text
text_id,infer
gist_12_3,치킨
gist_12_4,카페
```

Open that CSV if the printed match counts look wrong. The `text_id` column should look like `gist_12_3`, not like a path and not like `gist_12_3.jpg`.

`preds.txt` replaces an existing `preds.csv` in the same folder.

### JSONL

One JSON object per line. Pass the key that holds the image path and the key that holds the prediction. Those names are whatever your file uses.

```text
{"image_path": "GIST_Signboard/text_images/0/gist_12_3.jpg", "pred": "치킨"}
{"image_path": "GIST_Signboard/text_images/0/gist_12_4.jpg", "pred": "카페"}
```

The path key is handled the same way as a TXT path: only the file name becomes `text_id`. The tool writes `preds.csv` next to `preds.jsonl`, and replaces an existing file of that name.

## 2. Run

CSV:

```bash
python -m kstrbench.eval --dataset GIST_Signboard --pred preds.csv
```

TXT:

```bash
python -m kstrbench.eval --dataset GIST_Signboard --pred preds.txt
```

JSONL:

```bash
python -m kstrbench.eval --dataset GIST_Signboard --pred preds.jsonl --path-key image_path --infer-key pred
```

`--dataset` must be exactly one of `AIHub_Standardized`, `GIST_Signboard`, or `KAIST_Corrected`. `--path-key` and `--infer-key` are required for `.jsonl` and are ignored for the other two formats.

Run it once per subset. A file of GIST predictions is scored only against `GIST_Signboard`.

## 3. What is scored

The tool loads every `text_id` and `label` from that subset's label CSV. It then walks your inference rows.

- A row whose `text_id` is in the label CSV is **matched**. It receives the official `label` and two scores.
- A row whose `text_id` is not in the label CSV is **unmatched**. It stays in the output, with empty score columns. It is not averaged in.
- A label-CSV sample that never appears in your file is only counted. It is not added to the output, and it is not treated as a wrong prediction.

Word accuracy and jaso 1-NED are averages over matched rows only. If nothing matches, both are printed as `n/a`.

Spaces are removed before comparison, so `퀸 노래방` and `퀸노래방` match. Upper and lower case stay different, so `Cafe` and `cafe` do not match.

**Word accuracy** is the fraction of matched rows whose prediction equals the official label. `is_match` is `1` or `0`. One different character makes that row `0`.

**Jaso 1-NED** is a softer score for Korean. Each Hangul syllable is split into choseong, jungseong, and jongseong, and those units are compared with edit distance. `1` means the jamo sequence matches. A lower number means the prediction is farther from the label. A row can have `is_match` `0` and still have a high jaso score.

If a label contains `□`, that character may stand for zero or one predicted character and does not count against the model. It marks a glyph the annotator could not read.

For `GIST_Signboard` and `KAIST_Corrected`, the same two averages are also printed for each challenge tag. A matched row is included in a tag when that sample's label-CSV flag is true: `artistic`, `curve`, `sailent`, `incomplete`, `multi_oriented`, and `low_visibility`. One sample can belong to more than one tag, and then it counts in each of those averages. A tag with no matched row is printed as `n/a`. `AIHub_Standardized` has no challenge tags, so that block is omitted.

## 4. What is saved

Your original inference file is left as it is. New files are written beside it.

| You pass | Also written | Scored file |
| --- | --- | --- |
| `preds.csv` | nothing | `preds.eval.csv` |
| `preds.txt` | `preds.csv` | `preds.eval.csv` |
| `preds.jsonl` | `preds.csv` | `preds.eval.csv` |

`preds.csv` from a TXT or JSONL run has only `text_id` and `infer`. `preds.eval.csv` is the file to keep. It has the prediction, the official label, and the two scores:

```text
text_id,infer,label,is_match,jaso_1_ned
gist_12_3,치킨,치킨,1,1.000000
gist_12_4,친친,치킨,0,0.666667
not_in_gist,카페,,,
```

The third row was not in `GIST_Signboard`, so `label`, `is_match`, and `jaso_1_ned` are empty. If your CSV had other columns, they are copied into `preds.eval.csv` as well. Any `label` column you already had is replaced by the official label.

## 5. How to read the printout

```text
dataset           : GIST_Signboard
pred              : preds.txt
csv               : preds.csv
output            : preds.eval.csv
inference results : 3
matched           : 2
unmatched infer   : 1
unmatched dataset : 35244
accuracy          : 0.500000
jaso_1_ned        : 0.833333
challenges
  artistic         samples=1  ACC=1.000000  1-G_NED=1.000000
  curve            samples=1  ACC=0.000000  1-G_NED=0.666667
  sailent          samples=0  ACC=n/a       1-G_NED=n/a
  incomplete       samples=0  ACC=n/a       1-G_NED=n/a
  multi_oriented   samples=0  ACC=n/a       1-G_NED=n/a
  low_visibility   samples=0  ACC=n/a       1-G_NED=n/a
```

Read it from the top.

1. `pred` is the file you passed. `csv` is the file that was scored. For a CSV input these two are the same. For TXT and JSONL, `csv` is the converted file.
2. `output` is the scored CSV from the previous section.
3. `inference results` is the number of rows in your file. Here it is 3.
4. `matched` is how many of those rows were found in the subset. Here it is 2. `unmatched infer` is the rest. 2 + 1 = 3.
5. `unmatched dataset` is how many samples in the subset label CSV do not appear in your file. It is a coverage count. Those samples are not scored as errors.
6. `accuracy` is the mean of `is_match` over the 2 matched rows. `jaso_1_ned` is the mean of the jaso scores over those same rows.
7. `challenges` appears for GIST and KAIST. `samples` is how many matched rows carry that tag. `ACC` and `1-G_NED` are the means over those rows only. `samples=0` means none of your matched predictions have that tag.

`unmatched dataset` is often large. The label CSV holds the whole subset. `AIHub_Standardized` includes train, val, and test. `GIST_Signboard` and `KAIST_Corrected` include every text row in their CSV, not only one split. Predicting the test list alone leaves the other rows in this count. Check `matched` and `unmatched infer` first. Those two describe your file.

## If the result looks wrong

**`accuracy` is `n/a` and `matched` is 0.** No `text_id` was found in that subset. Open the converted CSV, or the `text_id` column of your CSV. The values should be ids such as `gist_12_3`. A full path, a `.jpg` suffix, or a different subset's ids will not match. Also check that `--dataset` is the subset those ids belong to.

**`unmatched infer` is large.** Those predictions use ids that are not in the label CSV. The same id check as above usually explains it.

**The command says the label CSV was not found.** `DATASET_DIR` does not contain that subset, or `AIHub_Standardized` has not been built yet.

**A TXT line is rejected.** Every non-empty line needs one tab between the image path and the prediction.

**A JSONL line is rejected.** Each line must be one JSON object, and both `--path-key` and `--infer-key` must be present on every line.
