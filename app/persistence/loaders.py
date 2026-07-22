import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from app.persistence.schemas.clinicals import ClinicalFile
from app.persistence.schemas.costs import CostFile
from app.persistence.schemas.economic_policy import EconomicPolicyFile
from app.persistence.schemas.mortality import MortalityFile
from app.persistence.schemas.simulation import SimulationFile
from app.persistence.schemas.utilities import UtilityFile

T = TypeVar("T")


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file from disk into a Python dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_typed_json(
    path: str | Path,
    parser: Callable[[dict[str, Any]], T],
) -> T:
    """
    Load JSON and convert it into a typed object using a parser.
    """
    raw = load_json(path)
    return parser(raw)


def parse_cost_file(data: dict) -> CostFile:
    return CostFile.model_validate(data)


def parse_utilities(data: dict) -> UtilityFile:
    # Utility value from Mazza et al. 2016
    # TODO: PSA with Miners ~0.64
    return UtilityFile.model_validate(data)


def parse_clinical(data: dict) -> ClinicalFile:
    return ClinicalFile.model_validate(data)


def parse_simulation(data: dict) -> SimulationFile:
    return SimulationFile.model_validate(data)


def parse_mortality(data: dict) -> MortalityFile:
    return MortalityFile.model_validate(data)


def parse_economic_policy(data: dict) -> EconomicPolicyFile:
    return EconomicPolicyFile.model_validate(data)
