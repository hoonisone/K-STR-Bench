from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kstrbench.step import Step


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(kw_only=True)
class DeletePathStep(Step):
    path: str
    missing_ok: bool = True
    must_be_under: str = ""
    protected_paths: list[str] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        target = _resolve(self.path)
        if self.must_be_under:
            root = _resolve(self.must_be_under)
            if target == root or root not in target.parents:
                raise ValueError(f"refusing to delete path outside {root}: {target}")
        for raw in self.protected_paths:
            protected = _resolve(raw)
            if target == protected or protected in target.parents or target in protected.parents:
                raise ValueError(f"refusing to delete protected path: {target}")

        existed = target.exists()
        if not existed:
            if not self.missing_ok:
                raise FileNotFoundError(f"path not found: {target}")
            kind = "missing"
        elif target.is_dir():
            shutil.rmtree(target)
            kind = "dir"
        elif target.is_file() or target.is_symlink():
            target.unlink()
            kind = "file"
        else:
            raise ValueError(f"unsupported path type: {target}")

        return {
            "ok": True,
            "path": str(target),
            "existed": existed,
            "deleted": existed,
            "kind": kind,
        }
