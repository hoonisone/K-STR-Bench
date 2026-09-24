from __future__ import annotations

from pathlib import Path

# One-off script: run as-is.
INPUT_FILE = Path(r"d:\AIHub\resources\config copy.txt")
# Set None to overwrite input file.
OUTPUT_FILE = None


def transform_line(line: str) -> str:
    raw = line.rstrip("\n")
    if ":" in raw:
        return raw.split(":", 1)[0].rstrip()
    return raw.rstrip()


def main() -> None:
    input_file = INPUT_FILE.expanduser().resolve()
    output_file = OUTPUT_FILE.expanduser().resolve() if OUTPUT_FILE else input_file

    if not input_file.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    converted = [transform_line(line) + "\n" for line in lines]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(converted)

    print(f"Done: {input_file} -> {output_file} ({len(converted)} lines)")


if __name__ == "__main__":
    main()

