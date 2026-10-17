"""Sun position from UTC time (NOAA solar calculator equations).

Accurate to about 0.01 degrees for 1900-2100; checked against pvlib's NREL
algorithm (largest difference 0.02 degrees). Refraction is ignored, which
only matters within a few degrees of the horizon.
"""
import numpy as np

# Bayfordbury Observatory (University of Hertfordshire), home of the UoH
# all-sky cameras.
LAT, LON = 51.7753, -0.0950


def sun_position(unix_seconds, lat=LAT, lon=LON):
    """Return (azimuth, zenith) in degrees; azimuth 0 = north, 90 = east."""
    t = np.asarray(unix_seconds, dtype=np.float64)
    jd = t / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.radians((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)
    m = np.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    centre = np.radians(
        np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
        np.sin(2 * m) * (0.019993 - 0.000101 * jc) + np.sin(3 * m) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = l0 + centre - np.radians(0.00569 + 0.00478 * np.sin(omega))
    obliq = np.radians(23 + (26 + (21.448 - jc *
                                   (46.815 + jc *
                                    (0.00059 - jc * 0.001813))) / 60) / 60 +
                       0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(obliq) * np.sin(app_long))
    y = np.tan(obliq / 2)**2
    eq_time = 4 * np.degrees(y * np.sin(2 * l0) - 2 * ecc * np.sin(m) +
                             4 * ecc * y * np.sin(m) * np.cos(2 * l0) -
                             0.5 * y * y * np.sin(4 * l0) -
                             1.25 * ecc * ecc * np.sin(2 * m))
    minutes = (t % 86400) / 60.0
    true_solar = (minutes + eq_time + 4 * lon) % 1440
    hour_angle = np.radians(true_solar / 4 - 180)
    phi = np.radians(lat)
    cos_z = (np.sin(phi) * np.sin(decl) +
             np.cos(phi) * np.cos(decl) * np.cos(hour_angle))
    zenith = np.arccos(np.clip(cos_z, -1, 1))
    azimuth = np.degrees(
        np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180
    return azimuth % 360, np.degrees(zenith)
