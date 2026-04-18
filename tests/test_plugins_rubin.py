from rubin_sim.phot_utils import Bandpass
from metroid.plugins.rubin import RubinBandpassProvider


def test_loads():
    """Test that the loads method of RubinBandPassProvider returns correct 
    result for valid cases.
    """
    provider = RubinBandpassProvider()
    bandpasses = provider.load('u')
    assert isinstance(bandpasses, dict)
    assert isinstance(bandpasses['u'], Bandpass)
