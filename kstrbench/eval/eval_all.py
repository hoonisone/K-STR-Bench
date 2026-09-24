from __future__ import annotations

from kstrbench.dataset_dir import AIHUB, GIST, KAIST, dataset_root, text_csv
from kstrbench.eval.steps.join_label_into_infer_csv import JoinLabelIntoInferCsvStep
from kstrbench.eval.steps.score_infer_csv import ScoreInferCsvStep
from kstrbench.eval.steps.summarize_infer_scores import SummarizeInferScoresStep
from kstrbench.step import Pipeline
_ROOT = dataset_root()
INFER_CSV_DIR = str(_ROOT / "infer_results" / "kor")
SUMMARY_CSV_PATH = str(_ROOT / "infer_result_summary.csv")

CSV_ENCODING = "utf-8-sig"
TEXT_ID_COL = "text_id"
LABEL_COL = "label"
INFER_COL = "infer"
IGNORE_BLANK = True

regimes = [
    "pretrained",
    "pretrained___synth_88_12",
    "scratch",
    "scratch___synth_88_12",
]

models = [
    "abinet",
    "busnet",
    "cdistnet",
    "cppd",
    "dan",
    "lister",
    "lpv",
    "maerec",
    "matrn",
    "mgp-str",
    "moran",
    "nrtr",
    "parseq",
    "roscanner",
    "sar",
    "smtr",
    "srn",
    "svtr1",
    "svtr2",
    "visionlan",
]

label_csv_by_dataset = {
    "aihub": str(text_csv(_ROOT, AIHUB)),
    "gist": str(text_csv(_ROOT, GIST)),
    "kaist": str(text_csv(_ROOT, KAIST)),
}

# A summary dataset is one or more scored CSVs. gist and kaist are merged so
# their challenge tags are measured on the combined set.
summary_dataset_sources = {
    "aihub": ["aihub"],
    "gist": ["gist"],
    "kaist": ["kaist"],
    "gist_kaist": ["gist", "kaist"],
}

challenges = [
    "KSTR_is_artistic",
    "KSTR_is_curve",
    "KSTR_is_sailent",
    "KSTR_is_incomplete",
    "KSTR_is_multi_oriented",
    "KSTR_is_low_visibility",
    # "is_noisy_background",
    # "is_stylized",
]

tag_csv_by_source = {
    "aihub": str(text_csv(_ROOT, AIHUB)),
    "gist": str(text_csv(_ROOT, GIST)),
    "kaist": str(text_csv(_ROOT, KAIST)),
}

pipelines = {}

join_label_steps = []
for regime in regimes:
    for model in models:
        for dataset, label_csv_path in label_csv_by_dataset.items():
            join_label_steps.append(
                JoinLabelIntoInferCsvStep(
                    name=f"join_label/{regime}/{model}/{dataset}",
                    label_csv_path=label_csv_path,
                    infer_csv_path=f"{INFER_CSV_DIR}\\{regime}\\{model}\\{dataset}.csv",
                    output_csv_path=f"{INFER_CSV_DIR}\\{regime}\\{model}\\{dataset}.csv",
                    text_id_col=TEXT_ID_COL,
                    label_col=LABEL_COL,
                    csv_encoding=CSV_ENCODING,
                )
            )

pipelines["join_label"] = Pipeline(
    name="join_label",
    steps=join_label_steps,
)


calculate_steps = []
for regime in regimes:
    for model in models:
        for dataset in label_csv_by_dataset:
            calculate_steps.append(
                ScoreInferCsvStep(
                    name=f"calculate/{regime}/{model}/{dataset}",
                    csv_path=f"{INFER_CSV_DIR}\\{regime}\\{model}\\{dataset}.csv",
                    output_csv_path=f"{INFER_CSV_DIR}\\{regime}\\{model}\\{dataset}.csv",
                    infer_col=INFER_COL,
                    label_col=LABEL_COL,
                    ignore_blank=IGNORE_BLANK,
                    csv_encoding=CSV_ENCODING,
                )
            )

pipelines["calculate"] = Pipeline(
    name="calculate",
    steps=calculate_steps,
)


summarize_steps = [
    SummarizeInferScoresStep(
        name="summarize",
        scored_csv_dir=INFER_CSV_DIR,
        output_csv_path=SUMMARY_CSV_PATH,
        regimes=regimes,
        models=models,
        dataset_sources=summary_dataset_sources,
        challenge_cols=challenges,
        tag_csv_by_source=tag_csv_by_source,
        text_id_col=TEXT_ID_COL,
        csv_encoding=CSV_ENCODING,
    )
]

pipelines["summarize"] = Pipeline(
    name="summarize",
    steps=summarize_steps,
)


if __name__ == "__main__":
    pipelines["join_label"].run()
    pipelines["calculate"].run()
    pipelines["summarize"].run()
