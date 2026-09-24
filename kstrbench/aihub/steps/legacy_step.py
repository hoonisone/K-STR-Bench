from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

from kstrbench.step import Step


@dataclass(kw_only=True)
class LegacyStep(Step):
    """Adapt an old AIHub step class (`__init__` + `run`) to `kstrbench.step.Step`."""

    impl_cls: Type[Any]
    init_args: Dict[str, Any] = field(default_factory=dict)

    def run(self) -> Dict[str, Any]:
        return self.impl_cls(**self.init_args).run()


def wrap(
    impl_cls: Type[Any],
    *,
    name: str = "",
    enabled: bool = True,
    **init_args: Any,
) -> LegacyStep:
    return LegacyStep(
        name=name or impl_cls.__name__,
        enabled=enabled,
        impl_cls=impl_cls,
        init_args=dict(init_args),
    )
