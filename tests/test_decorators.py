import subprocess

import pytest

from utils.decorators import deprecated, stable_hash


def test_deprecated_decorator_emits_warning():
    @deprecated("use new_function instead")
    def old_function(x, y):
        return x + y

    with pytest.warns(DeprecationWarning, match="use new_function"):
        result = old_function(2, 3)

    assert result == 5


# ── stable_hash ──────────────────────────────────────────────────────
#
# ``stable_hash`` replaces ``hash((seed, scenario.name))`` for deriving
# per-scenario PSA seeds. It MUST be identical across separate Python
# processes (Python's built-in ``hash()`` is randomized by PYTHONHASHSEED,
# which is why we need a stable replacement).


def test_stable_hash_is_deterministic_within_process():
    a = stable_hash(468498, "lifetime on_demand bayesian")
    b = stable_hash(468498, "lifetime on_demand bayesian")
    assert a == b


def test_stable_hash_differs_for_different_inputs():
    assert stable_hash(42, "a") != stable_hash(42, "b")
    assert stable_hash(42, "x") != stable_hash(7, "x")


def test_stable_hash_default_modulus_is_2_to_32():
    a = stable_hash(123, "scenario")
    assert 0 <= a < 2**32


def test_stable_hash_is_stable_across_python_processes():
    """Two independent Python invocations must agree — this is the
    whole point of the helper, and the regression that would re-break
    notebook reproducibility if anyone reintroduced ``hash()`` here."""
    snippet = (
        "from utils.decorators import stable_hash; "
        "print(stable_hash(468498, 'lifetime on_demand bayesian'))"
    )
    out1 = subprocess.run(
        ["python", "-c", snippet], capture_output=True, text=True, check=True
    ).stdout.strip()
    out2 = subprocess.run(
        ["python", "-c", snippet], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out1 == out2, (
        f"stable_hash is not cross-process stable: {out1!r} != {out2!r}"
    )
