"""
Lightweight step/pipeline primitives for local data processing workflows.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(kw_only=True)
class Step(ABC):
    name: str = ""
    enabled: bool = True

    @property
    def step_name(self) -> str:
        return self.name or self.__class__.__name__

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the step and return a result mapping."""


@dataclass
class Pipeline:
    steps: List[Step]
    name: str = "pipeline"

    def run(self) -> List[Dict[str, Any]]:
        if not self.steps:
            raise ValueError(f"Pipeline '{self.name}' has no steps.")

        total_steps = len(self.steps)
        results: List[Dict[str, Any]] = []

        print(f"=== Pipeline Start ({self.name}, steps: {total_steps}) ===")
        for index, step in enumerate(self.steps, start=1):
            if not step.enabled:
                print(f"[{index}/{total_steps}] Skip: {step.step_name}")
                continue

            print(f"\n[{index}/{total_steps}] Run: {step.step_name}")
            result = step.run()
            if result is None:
                result = {}
            if not isinstance(result, dict):
                result = {"result": result}

            results.append(
                {
                    "name": step.step_name,
                    "index": index,
                    "result": result,
                }
            )
            if result:
                print(f"[{step.step_name}] result: {result}")

        print(f"\n=== Pipeline Complete ({self.name}) ===")
        return results
