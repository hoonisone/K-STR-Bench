# Scene Text Recognition Dataset

## Dataset Path
`datasets/STR`

## What It Contains
- Text-unit crops (one crop per valid text instance).
- `label.txt` / `Label.txt` with `image_path<TAB>transcription` format.
- `pairs.json` mapping each crop to source image path and source annotation index.
- Optional analysis outputs under `analysis/` (character/group/length statistics).

## What You Can Do With It
- Train and evaluate scene text recognition (STR/OCR) models.
- Build language-specific subsets (for example, Korean-only filtering).
- Perform dataset diagnostics such as mismatch review and character distribution analysis.

## Key Characteristics
- Placeholder labels are excluded during crop generation.
- Chunked image storage (`images/<bucket>/<index>.jpg`) for scalable data handling.
- Includes mapping metadata for source-level traceability and debugging.
- Covers mixed-language and symbol-rich real-world signboard text.
