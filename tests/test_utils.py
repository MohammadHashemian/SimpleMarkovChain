# import pytest
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("__pytest__")


def test_s_graph_range():
    from utils import path_utils

    path = path_utils.get_project_root()
    assert str(path) == "/home/mohammad/projects/Thesis/hemophilia"
