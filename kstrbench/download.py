import argparse
from huggingface_hub import snapshot_download

from kstrbench.dataset_dir import dataset_root


REPO_ID = "hoonisone/K-STR-Bench"

# STR crops + label tables only. `text*.csv` would also pull text_old.csv.
# Image directories are released as tar archives (text_images.tar).
CORE_PATTERNS = [
    "**/text_images.tar",
    "**/text.csv",
    "**/text.v*.csv",
    "**/text.patch.v*.csv",
]


def download(mode: str, output_dir: str):
    if mode == "core":
        print("Downloading K-STR-Bench core dataset...")

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=CORE_PATTERNS,
        )

    elif mode == "all":
        print("Downloading full K-STR-Bench dataset...")

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=output_dir,
        )

    print(f"Download completed: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download K-STR-Bench from Hugging Face"
    )

    parser.add_argument(
        "--mode",
        choices=["core", "all"],
        default="core",
        help="core: text_images.tar and text.csv / text.v*.csv / text.patch.v*.csv; all: full dataset including scene image archives",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save the dataset. If omitted, use the DATASET_DIR environment variable.",
    )

    args = parser.parse_args()

    download(args.mode, str(dataset_root(args.output)))
