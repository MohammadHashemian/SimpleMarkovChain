# import pytest
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("__pytest__")


def test_costs_data():
    from utils.path_utils import get_project_root
    from persistence.loaders import load_typed_json, parse_cost_file
    from persistence.schemas.costs import CostFile

    file_path = get_project_root() / "data" / "economic.json"

    result = load_typed_json(file_path, parse_cost_file)

    # 1. Type check
    assert isinstance(result, CostFile)

    # 2. Structure checks
    assert len(result.currencies) > 0
    assert len(result.costs) > 0

    # 3. Deep field validation
    first_cost = result.costs[0]
    assert first_cost.item == "factor_viii"
    assert first_cost.pricing.per_unit["IRR"] > 0
    assert first_cost.assumption.iu_per_microgram > 0


def test_clinical_data():
    from utils.path_utils import get_project_root
    from persistence.loaders import load_typed_json, parse_clinical
    from persistence.schemas.clinicals import ClinicalFile

    file_path = get_project_root() / "data" / "clinical.json"

    result = load_typed_json(file_path, parse_clinical)

    # 1. Type check
    assert isinstance(result, ClinicalFile)


def test_utilities_data():
    from utils.path_utils import get_project_root
    from persistence.loaders import load_typed_json, parse_utilities
    from persistence.schemas.utilities import UtilityFile

    file_path = get_project_root() / "data" / "utilities.json"

    result = load_typed_json(file_path, parse_utilities)

    # 1. Type check
    assert isinstance(result, UtilityFile)


def test_simulation_data():
    from utils.path_utils import get_project_root
    from persistence.loaders import load_typed_json, parse_simulation
    from persistence.schemas.simulation import SimulationFile

    file_path = get_project_root() / "data" / "simulation.json"

    result = load_typed_json(file_path, parse_simulation)

    # 1. Type check
    assert isinstance(result, SimulationFile)


def test_mortality_data():
    from utils.path_utils import get_project_root
    from persistence.loaders import load_typed_json, parse_mortality
    from persistence.schemas.mortality import MortalityFile

    file_path = get_project_root() / "data" / "mortality.json"

    result = load_typed_json(file_path, parse_mortality)

    # 1. Type check
    assert isinstance(result, MortalityFile)
