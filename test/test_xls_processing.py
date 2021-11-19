from biopsias.form.xls_raw_preprocessing import assert_no_conflict

import pytest


# ya lo movere donde sea necesario
def test_assert_no_conflict():
    dicts = [
        {"a": 0, "b": 1, "c": 2, "d": 5},
        {"b": 1, "c": 2},
        {"a": 0, "b": 3, "c": 2},
    ]

    with pytest.raises(AssertionError):
        assert_no_conflict(*dicts)
