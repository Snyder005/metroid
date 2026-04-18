import os

from rubin_sim.phot_utils import Bandpass
from rubin_sim.data import get_data_dir


class RubinSimBandpassReader:

    def load(self, bands: list[str]) -> dict[str, Bandpass]:
        bandpasses = {}
        for band in bands:
            if not isinstance(band, str):
                raise TypeError("must be 'str'")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{band}.dat")
            bandpass = Bandpass()
            bandpass.read_throughput(filename)
            bandpasses[band] = bandpass

        return bandpasses
