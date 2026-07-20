import pytest
from sympy import symbols

pytestmark = pytest.mark.unit


def test_symbols_and_expression_string_forms():
    x, y = symbols("x y")
    assert str(x) == "x"
    assert str(y) == "y"
    assert str(x + y) == "x + y"
