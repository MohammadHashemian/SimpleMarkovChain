from dataclasses import dataclass
from typing import ClassVar

from app.persistence.loaders import (
    load_typed_json,
    parse_clinical,
    parse_cost_file,
    parse_economic_policy,
    parse_mortality,
    parse_simulation,
    parse_utilities,
)
from app.persistence.schemas.clinicals import ClinicalFile
from app.persistence.schemas.costs import CostFile
from app.persistence.schemas.economic_policy import EconomicPolicyFile
from app.persistence.schemas.mortality import MortalityFile
from app.persistence.schemas.simulation import SimulationFile
from app.persistence.schemas.utilities import UtilityFile
from utils.path_utils import get_project_root

_MORTALITY_FILE_BY_SOURCE: dict[str, str] = {
    "iran": "app/data/mortality_iran.json",
    "poland": "app/data/mortality.json",
    "default": "app/data/mortality.json",
}


@dataclass(frozen=True, slots=True)
class ModelContext:
    PROJECT_ROOT: ClassVar = get_project_root()

    simulation: SimulationFile
    clinical: ClinicalFile
    costs: CostFile
    economic_policy: EconomicPolicyFile
    utilities: UtilityFile
    mortality: MortalityFile

    @classmethod
    def load(cls) -> "ModelContext":
        root = cls.PROJECT_ROOT
        simulation = load_typed_json(
            root / "app/data/simulation.json", parse_simulation
        )
        mort_rel = _MORTALITY_FILE_BY_SOURCE.get(
            simulation.mortality.source, _MORTALITY_FILE_BY_SOURCE["default"]
        )
        return cls(
            simulation=simulation,
            clinical=load_typed_json(root / "app/data/clinical.json", parse_clinical),
            costs=load_typed_json(root / "app/data/economic.json", parse_cost_file),
            utilities=load_typed_json(root / "app/data/utilities.json", parse_utilities),
            mortality=load_typed_json(root / mort_rel, parse_mortality),
            economic_policy=load_typed_json(
                root / "app/data/economic_policy.json", parse_economic_policy
            ),
        )


# Annual bleeding rates – on-demand (mean, sd)
# (58.3, 26.9),  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
# (37.2, 19.9),  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
# (19.5, 15.0),  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
# (17.7, 11.7),  # Tagliaferri A et al. (26-55 y) 10.1160/TH14-05-0407
# (7.4, 9.5),  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
# (16.8, 10.0),  # Belgium
# (13.8, 12.6),  # France
# (12.2, 18.1),  # Germany
# (11.5, 11.5),  # Italy
# (7.0, 6.4),  # Spain
# (4.5, 0.7),  # Sweden
# (19.4, 10.6),  # UK
# (14.0, 12.3),  # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361 - First year
# (18.4, 14.3),  # Khair K et al. - Second year
# (15.8, 8.13),  # Khair K et al. - Third year
# (58.9, 16.6),  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
# (5.6, 1.83),  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
# (13.9, 4.47),  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
# (57.7, 24.6),  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
# (37.9, 33.08),  # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) 10.1007/s12185-018-02574-x
# (22.2, 7.0),  # Ying Liu (n=34; 4-18y, mean 12.2y) 10.1111/hae.14016
# (17.7, 9.3),  # B Warren (n:37, 2.5 up to 7.5y) 10.1182/bloodadvances.2019001311
# (17.69,9.25),  # Marilyn J. Manco-Johnson (<1.5y to 6y, n:65) 10.1056/NEJMoa067659
# (13, 9),  # Melissa Kern (n:15, pre target joint) 10.1016/j.jpeds.2004.06.082
# (24,12),  # Melissa Kern (n:15, post target joint) 10.1016/j.jpeds.2004.06.082
# (35.8,24.8),  # A. Tagliaferri (n: 83, median 23.6, 10-72y) 10.1111/j.1365-2516.2008.01791.x
# (21.42, 9.59),  # Aznar (n: 15, 26-47 mean 35.6) 10.1111/vox.12066
# (35.7,22.2),  # von Drygalski (26 adults, mean 42.8 y) 10.1056/NEJMoa2209226
# (22.2, 7.0),  # Liu, ODT group (n=18, age 12.4 SD 4.1) 10.1111/hae.14016

# Annual bleeding rates – prophylaxis (mean, sd)
# (2.5, 4.6),  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
# (2.5, 4.7),  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
# (2.6, 2.2),  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
# (4.5, 7.1),  # Tagliaferri A et al. (26-55 y) 10.1160/TH14-05-0407
# (2.1, 2.1),  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
# (4.1, 6.9),  # Belgium
# (8.0, 9.4),  # France
# (4.5, 5.3),  # Germany
# (2.8, 4.7),  # Italy
# (2.7, 2.5),  # Spain
# (1.9, 2.9),  # Sweden
# (5.8, 7.0),  # UK
# (3.5,4.3),  # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361 - First year
# (3.3, 4.1),  # Khair K et al. - Second year
# (3.7, 3.9),  # Khair K et al. - Third year
# (3.0, 5.9),  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
# (1.86, 1.52),  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
# (4.8, 5.0),  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
# (4.3, 6.5),  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
# (8.9, 19.61),  # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) 10.1007/s12185-018-02574-x
# (3.5, 2.1),  # Beth Boulden Warren (n:37, 2.5 up to 18y) - Early proph group  # noqa: E501
# (6.2, 5.3),  # Beth Boulden Warren - Post-proph group 10.1182/bloodadvances.2019001311
# (3.27, 6.24),  # Marilyn J. Manco-Johnson (<1.5y to 6y, n:65) 10.1056/NEJMoa067659
# (4.2, 3.7),  # A. Tagliaferri (n: 83, median 23.6, 10-72y) 10.1111/j.1365-2516.2008.01791.x
# (1.82, 2.87),  # Aznar (n: 15, 26-47 mean 35.6) 10.1111/vox.12066
# (3.2, 5.4),  # von Drygalski (133, ≥12 y, mean 33.9 y) 10.1056/NEJMoa2209226


# Treatment dosing (Intensity Regimen Protocol)
# IR Protocol: 50 IU/kg twice weekly
# ir_prophylaxis_weekly_dose_ui: int = 25 * 2
# Standard Protocol: 25 IU/kg three times weekly
# standard_prophylaxis_weekly_dose_ui: int = 25 * 3
# Bleeding treatment doses (per bleeding event, before body weight adjustment)
# bleeding_dose_ui: int = 30 * 4  # Standard bleeding dose: 30 IU/kg × 4 injections
# joint_bleeding_dose_ui: int = 30 * 2  # Joint bleeding dose: 30 IU/kg × 2 injections
# lt_bleeding_dose_ui: int = 550  # Life-threatening bleeding dose: 550 IU/kg
