# Signboard Text Detection and Recognition Dataset

## Dataset Path
`datasets/signboard_rec`

## What It Contains
- Cropped signboard images generated from the original dataset.
- `Label.txt` where each line stores a crop path and text-region annotations in JSON.
- `pairs.json` mapping each crop to its source image/sample index.
- Error logs (`errors.jsonl`) for failed crop samples, if any.

## What You Can Do With It
- Train signboard-level text detection models on focused signboard crops.
- Run end-to-end signboard text pipelines (detect text regions and recognize text).
- Analyze text assignment quality from source annotations via `pairs.json`.

## Key Characteristics
- Crops include signboard area plus assigned text polygons.
- Text assignment is based on polygon overlap rules from source annotations.
- Supports traceability to source data through mapping metadata.
- Better text-region density than full-scene images for signboard-centric tasks.
