"""Fisheye camera model: sun direction -> pixel in the 224x224 stored frames."""
import json
from dataclasses import asdict, dataclass

import numpy as np

from sunpos import sun_position


@dataclass
class CameraModel:
    """r = f * zen + k * zen^3 pixels from the centre (zen in radians).

    ``north`` is the image direction of north in degrees (clockwise from up),
    ``mirror`` is -1 when east is on the left (as in the UoH frames).
    """
    u0: float
    v0: float
    f: float
    k: float
    north: float
    mirror: int = -1

    def project(self, azimuth, zenith):
        """Pixel ``[..., 2]`` (x, y) of directions given in degrees."""
        az, zen = np.radians(azimuth), np.radians(zenith)
        r = self.f * zen + self.k * zen**3
        a = az - np.radians(self.north)
        return np.stack([
            self.u0 + self.mirror * r * np.sin(a), self.v0 - r * np.cos(a)
        ], -1)

    def sun_pixel(self, unix_seconds):
        return self.project(*sun_position(unix_seconds))

    def save(self, path, **extra):
        with open(path, 'w') as f:
            json.dump({**asdict(self), **extra}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


# Fitted on the 167 frames of figures/uoh_2016-06-15.gif (median error
# 0.4 px); 2016-05-04 and 2018-03-16 frames land within 3 and 1 px.
JUNE_2016 = CameraModel(u0=111.542, v0=97.964, f=77.584, k=-5.771,
                        north=13.506, mirror=-1)
