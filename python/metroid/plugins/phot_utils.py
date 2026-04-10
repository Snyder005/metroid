import os

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

def calculate_adu( 
        magnitude: float, 
        band: str, 
        exptime: float, 
        effarea: float, 
        pixel_scale: float, 
        gain: float = 1.0,
    ) -> int:

    # Get photo params
    phot_params = photUtils.PhotometricParameters(
        exptime=exptime,
        nexp=1,
        effarea=effarea,
        gain=gain,
        platescale=pixel_scale,
    )

    # Get bandpass
    filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{band}.dat")
    bandpass = photUtils.Bandpass()
    bandpass.read_throughput(filename)

    # Define SED
    sed = photUtils.Sed()
    sed.set_flat_sed()

    adu_m0 = sed.calc_adu(bandpass, phot_params=phot_params)
    adu = adu_m0*(10.0**(-magnitude/2.5))

    return adu
