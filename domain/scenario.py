from typing import Any, Dict, Generic, TypeVar, List
from pydantic import BaseModel, Field
from analysis.psa.models import ParameterSet
from analysis.psa.parameters import Parameter
from domain.enums import Regime
from dataclasses import dataclass
import copy

T = TypeVar("T")


class Scenario(BaseModel):
    name: str
    regime: Regime
    description: str | None = None

    overrides: Dict[str, Any] = Field(default_factory=dict)

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
    inputs: List[T]
