import copy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameters import Parameter
from app.domain.enums import Regime

T = TypeVar("T")


class Scenario(BaseModel):
    name: str
    regime: Regime
    description: str | None = None

    overrides: dict[str, Any] = Field(default_factory=dict)

    def apply_overrides(self, base: ParameterSet) -> ParameterSet:
        """
        Safely apply overrides to a ParameterSet.
        Keeps Parameter objects intact.
        """

        # Shallow copy of dataclass (keeps Parameter objects)
        new_params = copy.copy(base)

        for key, value in self.overrides.items():

            if not hasattr(new_params, key):
                raise ValueError(f"Override key '{key}' not found in ParameterSet")

            # 🔒 Enforce type safety (critical for PSA)
            if not isinstance(value, Parameter):
                raise TypeError(
                    f"Override for '{key}' must be a Parameter, got {type(value)}"
                )

            setattr(new_params, key, value)

        return new_params


@dataclass
class ScenarioBundle(Generic[T]):
    scenario: Scenario
    inputs: list[T]
