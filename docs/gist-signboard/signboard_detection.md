# Signboard Detection Dataset

## Dataset Path
`datasets/signboard_det`

## What It Contains
- Full scene images collected in real urban environments.
- Polygon annotations for signboard regions.
- Polygon annotations for text regions in the same image.
- Per-image label records in `Label.txt` / `label.txt` format.

## What You Can Do With It
- Train and evaluate signboard localization/detection models.
- Benchmark detection robustness under perspective distortion, occlusion, and clutter.
- Build preprocessing pipelines that detect signboard candidates before recognition.

## Key Characteristics
- Multi-signboard scenes: one image can include multiple signboards.
- Mixed geometry: annotations are polygons, not only axis-aligned boxes.
- Real-world variation: different lighting, scales, rotations, and backgrounds.
- Includes both signboard and text annotations, enabling downstream task generation.
