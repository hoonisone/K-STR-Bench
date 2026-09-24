from __future__ import annotations

import hashlib
import os
from pathlib import Path

# 라인 매핑 기준 파일
OLD_LIST_FILE = Path(r"d:\AIHub\resources\AIHub_organized\metadata\unmatched_file_list copy.txt")
NEW_LIST_FILE = Path(r"d:\AIHub\resources\AIHub_organized\metadata\unmatched_file_list copy 2.txt")

# 실제 경로가 존재하는 루트
DATASET_ROOT = Path(r"d:\AIHub\datasets\AIHub_organized")

# 옵션
DRY_RUN = False
SKIP_MISSING_SOURCE = True


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"dataset root not found: {DATASET_ROOT}")

    old_lines = _read_lines(OLD_LIST_FILE)
    new_lines = _read_lines(NEW_LIST_FILE)

    if len(old_lines) != len(new_lines):
        raise ValueError(
            f"line count mismatch: old={len(old_lines)} vs new={len(new_lines)}"
        )

    pairs: list[tuple[Path, Path]] = []
    for old, new in zip(old_lines, new_lines):
        old_rel = old.strip().replace("\\", "/")
        new_rel = new.strip().replace("\\", "/")
        if not old_rel:
            continue
        src = (DATASET_ROOT / old_rel).resolve()
        dst = (DATASET_ROOT / new_rel).resolve()
        if src == dst:
            continue
        pairs.append((src, dst))

    # 중복/충돌 체크
    src_set: set[Path] = set()
    dst_set: set[Path] = set()
    for src, dst in pairs:
        if src in src_set:
            raise ValueError(f"duplicate source path in mapping: {src}")
        if dst in dst_set:
            raise ValueError(f"duplicate destination path in mapping: {dst}")
        src_set.add(src)
        dst_set.add(dst)

    # 1단계: src -> tmp (깊은 경로 먼저)
    phase1: list[tuple[Path, Path]] = []
    for src, _dst in sorted(pairs, key=lambda x: len(x[0].parts), reverse=True):
        if not src.exists():
            if SKIP_MISSING_SOURCE:
                continue
            raise FileNotFoundError(f"source path not found: {src}")
        tmp_name = f"__tmp_rename__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:12]}"
        tmp = src.parent / tmp_name
        phase1.append((src, tmp))

    # 2단계: tmp -> dst (얕은 경로 먼저)
    src_to_tmp = {src: tmp for src, tmp in phase1}
    phase2: list[tuple[Path, Path]] = []
    for src, dst in sorted(pairs, key=lambda x: len(x[1].parts)):
        tmp = src_to_tmp.get(src)
        if tmp is None:
            continue
        phase2.append((tmp, dst))

    if DRY_RUN:
        print(f"[DRY_RUN] dataset_root={DATASET_ROOT}")
        print(f"[DRY_RUN] phase1_count={len(phase1)}")
        print(f"[DRY_RUN] phase2_count={len(phase2)}")
        return

    for src, tmp in phase1:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, tmp)

    for tmp, dst in phase2:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)

    print(
        f"Done: dataset_root={DATASET_ROOT}, "
        f"renamed={len(phase2)} pair(s)"
    )


if __name__ == "__main__":
    main()

