from __future__ import annotations

import argparse

import pytest

from scripts.train_finn_model import _parse_hidden_dims


def test_parse_hidden_dims_accepts_comma_separated_positive_integers():
    assert _parse_hidden_dims("32,64,128") == (32, 64, 128)
    assert _parse_hidden_dims(" 16 , 32 ") == (16, 32)


@pytest.mark.parametrize("value", ["", "64,", "0,64", "-1,64", "wide"])
def test_parse_hidden_dims_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hidden_dims(value)

