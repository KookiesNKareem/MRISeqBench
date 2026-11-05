"""Fixed-duration trapezoids that stretch only when the moment needs longer."""

import numpy as np
from pypulseq import calc_duration, make_trapezoid


def trapezoid(channel, *, duration=None, system=None, **kwargs):
    if duration is None or "area" not in kwargs:
        return make_trapezoid(channel, duration=duration, system=system, **kwargs)
    try:
        return make_trapezoid(channel, duration=duration, system=system, **kwargs)
    except (AssertionError, ValueError):
        shortest = calc_duration(make_trapezoid(channel, system=system, **kwargs))
        raster = 1e-5
        rounded = float(np.ceil(shortest / raster - 1e-9) * raster)
        try:
            return make_trapezoid(channel, duration=rounded, system=system, **kwargs)
        except (AssertionError, ValueError):
            # Some minimum-duration boundaries round one ULP too low.
            return make_trapezoid(
                channel,
                duration=float(np.nextafter(rounded, np.inf)),
                system=system,
                **kwargs,
            )
