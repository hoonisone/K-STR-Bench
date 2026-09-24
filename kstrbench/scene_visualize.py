from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from kstrbench.dataset_dir import AIHUB, GIST, KAIST, TEXT_CSV, dataset_root, subset_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "gist"
SCENE_ID = "gist_136"
SCENE_ID = "gist_767"
SCENE_ID = "gist_1743"

DATASET_NAME = "aihub"
SCENE_ID = "aihub_174826"
SCENE_ID = "aihub_174456"
SCENE_ID = "aihub_61090"


BOX_COLOR = (0, 255, 0)
LINE_WIDTH = 7
CSV_ENCODING = "utf-8-sig"
OUT_DIR = REPO_ROOT / "visualized"

DATASETS = {
    "aihub": {"subset": AIHUB, "scene_csv": "scene.csv"},
    "gist": {"subset": GIST, "scene_csv": "scene.csv"},
    "kaist": {"subset": KAIST, "scene_csv": "scene.csv"},
}


def resolve_dataset(name: str) -> tuple[Path, dict]:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "aihub_standardized": "aihub",
        "gist_signboard": "gist",
        "kaist_corrected": "kaist",
    }
    key = aliases.get(key, key)
    spec = DATASETS.get(key)
    if spec is None:
        raise ValueError(f"Unknown dataset: {name}. Use aihub, gist, or kaist.")

    root = dataset_root()
    folder = subset_dir(root, spec["subset"])
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    spec = {**spec, "text_csv": TEXT_CSV[spec["subset"]]}
    return folder, spec


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_points(raw: str) -> list[tuple[int, int]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        return []
    points: list[tuple[int, int]] = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((int(float(item[0])), int(float(item[1]))))
    return points


def find_scene(rows: list[dict[str, str]], scene_id: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("scene_id") or "").strip() == scene_id:
            return row
    raise KeyError(f"scene_id not found: {scene_id}")


def resolve_image_path(folder: Path, scene: dict[str, str]) -> Path:
    tried: list[str] = []
    for key in ("image_path", "origin_image_path"):
        rel = str(scene.get(key) or "").strip().replace("\\", "/")
        if not rel:
            continue
        path = Path(rel) if Path(rel).is_absolute() else folder / rel
        if path.is_file():
            return path
        tried.append(f"{key}={path}")
    raise FileNotFoundError(
        f"Scene image not found for scene_id={scene.get('scene_id')}. Tried: "
        + "; ".join(tried)
    )


def main() -> None:
    folder, spec = resolve_dataset(DATASET_NAME)
    scene = find_scene(read_csv(folder / spec["scene_csv"]), SCENE_ID)
    image_path = resolve_image_path(folder, scene)

    texts = [
        row
        for row in read_csv(folder / spec["text_csv"])
        if str(row.get("scene_id") or "").strip() == SCENE_ID
    ]

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    drawn = 0
    for row in texts:
        points = parse_points(row.get("points") or "")
        if len(points) < 2:
            continue
        draw.polygon(points, outline=BOX_COLOR, width=LINE_WIDTH)
        drawn += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / image_path.name
    image.save(out_path)
    print(f"{SCENE_ID}: {drawn} boxes -> {out_path}")


if __name__ == "__main__":
    main()
