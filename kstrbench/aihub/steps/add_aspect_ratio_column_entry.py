from __future__ import annotations

from typing import Any

from .add_arithmetic_column_entry import AddArithmeticColumnStep, run, run_from_config


class AddAspectRatioColumnStep(AddArithmeticColumnStep):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

