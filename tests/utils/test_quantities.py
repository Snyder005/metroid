from astropy import units as u
import pytest
from metroid.utils import quantities as q


def test_quantity_spec_creation():
    spec = q.QuantitySpec("area", u.m**2)

    assert spec.name == "area"
    assert spec.default == (u.m**2)
    assert spec.typical_range is None


@pytest.mark.parametrize(
    "annotation,expected_spec",
    {
        (q.Area, q.AREA),
        (None, None),
        (q.Area | None, q.AREA),
    },
)
def test_extract_spec(annotation, expected_spec):
    spec = q.extract_spec(annotation)

    assert spec == expected_spec
