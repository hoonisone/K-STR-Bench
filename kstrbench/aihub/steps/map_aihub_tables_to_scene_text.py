from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..columns import (
    ORIGIN_DATASET_AIHUB,
    SCENE_ID_PREFIX_AIHUB,
    SceneCol,
    TextCol,
)
from kstrbench.step import Step


def _parse_bbox(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, tuple)):
        return list(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def bbox_to_points(raw: Any) -> str:
    parsed = _parse_bbox(raw)
    if not isinstance(parsed, list) or not parsed:
        return ""

    if len(parsed) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in parsed):
        x, y, w, h = (float(v) for v in parsed)
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        return json.dumps(points, ensure_ascii=False)

    if len(parsed) == 4 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in parsed):
        points = [[float(v[0]), float(v[1])] for v in parsed]
        return json.dumps(points, ensure_ascii=False)

    return ""


def make_scene_id(image_id: Any, prefix: str = SCENE_ID_PREFIX_AIHUB) -> str:
    raw = str(image_id).strip()
    if not raw:
        return ""
    if prefix and (raw == prefix or raw.startswith(f"{prefix}_")):
        return raw
    return f"{prefix}_{raw}" if prefix else raw


def make_text_id(scene_id: str, text_index: Any) -> str:
    index = str(text_index).strip()
    if not scene_id or not index:
        return ""
    return f"{scene_id}_{index}"


@dataclass(kw_only=True)
class MapAihubTablesToSceneTextStep(Step):
    in_image_csv_file: str
    in_annotation_csv_file: str
    out_scene_csv_file: str
    out_text_csv_file: str
    origin_dataset: str = ORIGIN_DATASET_AIHUB
    scene_id_prefix: str = SCENE_ID_PREFIX_AIHUB
    csv_encoding: str = "utf-8-sig"
    show_progress: bool = True

    def run(self) -> dict[str, Any]:
        in_image_csv = Path(self.in_image_csv_file).expanduser().resolve()
        in_ann_csv = Path(self.in_annotation_csv_file).expanduser().resolve()
        out_scene_csv = Path(self.out_scene_csv_file).expanduser().resolve()
        out_text_csv = Path(self.out_text_csv_file).expanduser().resolve()

        if not in_image_csv.is_file():
            raise FileNotFoundError(f"in_image_csv_file not found: {in_image_csv}")
        if not in_ann_csv.is_file():
            raise FileNotFoundError(f"in_annotation_csv_file not found: {in_ann_csv}")

        scene_count = self._write_scene_csv(in_image_csv, out_scene_csv)
        text_count = self._write_text_csv(in_ann_csv, out_text_csv)
        return {
            "ok": True,
            "in_image_csv_file": str(in_image_csv),
            "in_annotation_csv_file": str(in_ann_csv),
            "out_scene_csv_file": str(out_scene_csv),
            "out_text_csv_file": str(out_text_csv),
            "scene_row_count": scene_count,
            "text_row_count": text_count,
            "origin_dataset": self.origin_dataset,
            "scene_id_prefix": self.scene_id_prefix,
        }

    def _write_scene_csv(self, in_csv: Path, out_csv: Path) -> int:
        with in_csv.open("r", encoding=self.csv_encoding, newline="") as src:
            reader = csv.DictReader(src)
            source_fields = list(reader.fieldnames or [])
            extra_fields = [
                name
                for name in source_fields
                if name
                not in {
                    "image_id",
                    "image_file",
                    "label_file",
                    SceneCol.SCENE_ID.value,
                    SceneCol.IMAGE_PATH.value,
                    SceneCol.ORIGIN_DATASET.value,
                    SceneCol.ORIGIN_IMAGE_PATH.value,
                }
            ]
            header = [
                SceneCol.SCENE_ID.value,
                SceneCol.ORIGIN_DATASET.value,
                SceneCol.IMAGE_PATH.value,
                SceneCol.ORIGIN_IMAGE_PATH.value,
                SceneCol.WIDTH.value,
                SceneCol.HEIGHT.value,
                SceneCol.DATA_CAPTURED_DATE.value,
            ]
            for name in extra_fields:
                if name not in header:
                    header.append(name)

            out_csv.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with out_csv.open("w", encoding=self.csv_encoding, newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    scene_id = make_scene_id(row.get("image_id", ""), self.scene_id_prefix)
                    mapped = {
                        SceneCol.SCENE_ID.value: scene_id,
                        SceneCol.ORIGIN_DATASET.value: self.origin_dataset,
                        SceneCol.IMAGE_PATH.value: row.get("image_file", row.get(SceneCol.IMAGE_PATH.value, "")),
                        SceneCol.ORIGIN_IMAGE_PATH.value: row.get(SceneCol.ORIGIN_IMAGE_PATH.value, ""),
                        SceneCol.WIDTH.value: row.get("width", ""),
                        SceneCol.HEIGHT.value: row.get("height", ""),
                        SceneCol.DATA_CAPTURED_DATE.value: row.get("data_captured_date", ""),
                    }
                    for name in extra_fields:
                        mapped[name] = row.get(name, "")
                    writer.writerow(mapped)
                    count += 1
            return count

    def _write_text_csv(self, in_csv: Path, out_csv: Path) -> int:
        with in_csv.open("r", encoding=self.csv_encoding, newline="") as src:
            reader = csv.DictReader(src)
            source_fields = list(reader.fieldnames or [])
            dropped = {
                "image_id",
                "inner_ann_id",
                "annotation_id",
                "text",
                "bbox",
                TextCol.TEXT_ID.value,
                TextCol.SCENE_ID.value,
                TextCol.TEXT_INDEX.value,
                TextCol.ORIGIN_DATASET.value,
                TextCol.LABEL.value,
                TextCol.POINTS.value,
            }
            extra_fields = [name for name in source_fields if name not in dropped]
            header = [
                TextCol.TEXT_ID.value,
                TextCol.SCENE_ID.value,
                TextCol.TEXT_INDEX.value,
                TextCol.ORIGIN_DATASET.value,
                TextCol.LABEL.value,
                TextCol.POINTS.value,
            ]
            for name in extra_fields:
                if name not in header:
                    header.append(name)

            out_csv.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with out_csv.open("w", encoding=self.csv_encoding, newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    scene_id = make_scene_id(row.get("image_id", ""), self.scene_id_prefix)
                    text_index = str(row.get("inner_ann_id", "")).strip()
                    mapped = {
                        TextCol.TEXT_ID.value: make_text_id(scene_id, text_index),
                        TextCol.SCENE_ID.value: scene_id,
                        TextCol.TEXT_INDEX.value: text_index,
                        TextCol.ORIGIN_DATASET.value: self.origin_dataset,
                        TextCol.LABEL.value: row.get("text", row.get(TextCol.LABEL.value, "")),
                        TextCol.POINTS.value: bbox_to_points(row.get("bbox")),
                    }
                    for name in extra_fields:
                        mapped[name] = row.get(name, "")
                    writer.writerow(mapped)
                    count += 1
            return count
