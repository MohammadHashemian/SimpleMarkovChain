import pytest

from utils.decorators import deprecated


def test_deprecated_decorator_emits_warning():
    @deprecated("use new_function instead")
    def old_function(x, y):
        return x + y

    with pytest.warns(DeprecationWarning, match="use new_function"):
        result = old_function(2, 3)

    assert result == 5
