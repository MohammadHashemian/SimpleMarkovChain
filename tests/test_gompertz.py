# import pytest
import logging
import sys
import pytest
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("__pytest__")


def test_s_graph_range():
    from utils.math import cal_body_weight

    # Valid range
    ages_in_week = [int(i) for i in np.linspace(0, 5200, 5201).tolist()]
    [cal_body_weight(age) for age in ages_in_week]

    # Invalid range
    with pytest.raises(ValueError) as exc_info:
        ages_in_week = [int(i) for i in np.linspace(0, 5201, 5202).tolist()]
        [cal_body_weight(age) for age in ages_in_week]
        assert "Week must be an integer" in str(exc_info.value)
