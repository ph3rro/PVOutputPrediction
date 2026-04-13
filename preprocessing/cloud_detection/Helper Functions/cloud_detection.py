import os
import numpy as np
from Sun_position import sun_position

"""
Created on December 2024
Revised onMarch 11 2026
@author: Lama EL Halabi
"""

def load_clear_sky_library(clear_sky_directory_path, year):
    """
    Load year-specific clear-sky-library arrays.

    Expected files:
        csl_images_{year}.npy
        csl_times_{year}.npy
        csl_sun_center_{year}.npy
    """
    csl_images = np.load(
        os.path.join(clear_sky_directory_path, f"csl_images_{year}.npy"),
        allow_pickle=True,
    )
    csl_times = np.load(
        os.path.join(clear_sky_directory_path, f"csl_times_{year}.npy"),
        allow_pickle=True,
    )
    csl_sun_center = np.load(
        os.path.join(clear_sky_directory_path, f"csl_sun_center_{year}.npy"),
        allow_pickle=True,
    )
    return csl_times, csl_images, csl_sun_center



def _safe_nrbr(img):
    """
    Compute the normalized red-blue ratio (NRBR) for each pixel:
        NRBR = (R - B) / (R + B)

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        2D float32 array of shape (H, W) containing pixel-wise NRBR values.
        Pixels with zero denominator are assigned value 0.
    """
    r = img[:, :, 0].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    denom = r + b

    nrbr = np.zeros_like(r, dtype=np.float32)
    valid = denom != 0
    nrbr[valid] = (r[valid] - b[valid]) / denom[valid]
    return nrbr



def _disk_mask(shape, center, radius):
    """
    Create a binary disk mask.

    Parameters
    ----------
    shape : tuple
        Image shape, typically (H, W, C) or (H, W).
    center : tuple
        Disk center as (row, col).
    radius : float or int
        Disk radius.

    Returns
    -------
    np.ndarray
        2D uint8 array of shape (H, W), with value 1 inside the disk
        and 0 outside.
    """
    rows, cols = np.indices(shape[:2])
    return (((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius ** 2).astype(np.uint8)

def _ring_mask(shape, center, inner_radius, outer_radius):
    """
    Create a binary ring mask.

    Returns a 2D uint8 array of shape (H, W), with 1 between the two radii.
    """
    rows, cols = np.indices(shape[:2])
    dist2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return ((dist2 >= inner_radius ** 2) & (dist2 <= outer_radius ** 2)).astype(np.uint8)

def _cloud_fraction(cloud_mask, support_mask):
    """
    Compute the fraction of cloudy pixels inside a support region.

    Parameters
    ----------
    cloud_mask : np.ndarray
        Binary cloud mask.
    support_mask : np.ndarray
        Binary mask defining the region over which cloudiness is measured.

    Returns
    -------
    float
        Fraction of cloudy pixels in the support region. Returns 0.0 if the
        support region is empty.
    """
    support = support_mask > 0
    n = np.count_nonzero(support)
    if n == 0:
        return 0.0
    return np.count_nonzero((cloud_mask > 0) & support) / n


def cloud_detection_modified(
    time,
    image,
    clear_sky_directory_path,
    year,
    twbs_th,
    fixed_th,
    cloudiness_th,
    circ_outer_r,
    sun_cloudiness_th,
    min_cloudiness_merge=0.045,
    circ_inner_r=None,
    sky_center=(29, 30),
    sky_radius=29,
):
    """
    Cloud detection using the modified threshold-with-subtraction method.

    Logic
    -----
    1. Build:
       - TWBS mask from dNRBR = |NRBR(image) - NRBR(clear_sky)|
       - fixed-threshold mask from NRBR(image)

    2. Compute overall cloudiness from the TWBS mask.

    3. Regimes:
       - low cloudiness: use TWBS mask
       - high cloudiness: use fixed-threshold mask
       - intermediate cloudiness:
           * build merged mask = TWBS OR fixed-threshold
           * compute sun_cloudiness on a circumsolar rim
           * outside the circumsolar disk: use merged mask
           * inside the circumsolar disk:
               - use TWBS if sun_cloudiness < sun_cloudiness_th
               - else use fixed-threshold

    Parameters
    ----------
    time : datetime-like
        Timestamp of the sky image.
    image : np.ndarray
        Input sky image of shape (H, W, 3).
    clear_sky_directory_path : str
        Path to the clear-sky library directory.
    year : int
        Clear-sky library year to load.
    twbs_th : float
        Threshold on dNRBR.
    fixed_th : float
        Fixed NRBR threshold.
    cloudiness_th : float
        Upper threshold separating intermediate and high-cloudiness regimes.
    circ_outer_r : int
        Outer radius of the circumsolar disk.
    sun_cloudiness_th : float
        Threshold on rim cloudiness used to decide whether the sun is under cloud.
    min_cloudiness_merge : float, optional
        Lower threshold separating low and intermediate cloudiness regimes.
    circ_inner_r : int or None, optional
        Inner radius of the circumsolar measurement rim.
        If None, uses max(circ_outer_r - 2, 0).
    sky_center : tuple, optional
        Center of the valid sky disk.
    sky_radius : int, optional
        Radius of the valid sky disk.

    Returns
    -------
    cloud_cover : float
        Final cloud fraction over the valid sky region.
    cloud_mask_tw : np.ndarray
        TWBS cloud mask.
    cloud_mask_twf : np.ndarray
        Final cloud mask.
    cloud_mask_f : np.ndarray
        Fixed-threshold cloud mask.
    circ_sun_disk_mask : np.ndarray
        Circumsolar disk mask used for inside/outside region handling.
    sun_cloudiness : float
        Cloud fraction measured on the circumsolar rim.
    """

    if circ_inner_r is None:
        circ_inner_r = max(circ_outer_r - 2, 0)

    _, csl_image, csl_sun_center = load_clear_sky_library(
        clear_sky_directory_path=clear_sky_directory_path,
        year=year,
    )

    csl_sun_center_x = csl_sun_center[:, 0]
    csl_sun_center_y = csl_sun_center[:, 1]

    # Sun position in the input image
    sun_center_x, sun_center_y = sun_position(time, image)

    # Valid sky region
    sky_mask = _disk_mask(image.shape, sky_center, sky_radius)

    # Circumsolar disk: region treated separately in the final mask
    circ_sun_disk_mask = (
        _disk_mask(image.shape, (sun_center_x, sun_center_y), circ_outer_r) * sky_mask
    ).astype(np.uint8)

    # Circumsolar rim: region used only to measure sun cloudiness
    circ_sun_rim_mask = (
        _ring_mask(
            image.shape,
            (sun_center_x, sun_center_y),
            inner_radius=circ_inner_r,
            outer_radius=circ_outer_r,
        ) * sky_mask
    ).astype(np.uint8)

    outside_circ_mask = ((sky_mask > 0) & (circ_sun_disk_mask == 0)).astype(np.uint8)

    # Match clear-sky image by closest sun position
    dist_sun_center = np.sqrt(
        (csl_sun_center_x - sun_center_x) ** 2
        + (csl_sun_center_y - sun_center_y) ** 2
    )
    match_idx = np.argmin(dist_sun_center)
    match_csl_image = csl_image[match_idx]

    # NRBR and subtraction
    nrbr_orig = _safe_nrbr(image)
    nrbr_cs = _safe_nrbr(match_csl_image)
    d_nrbr = np.abs(nrbr_orig - nrbr_cs)

    # TWBS mask
    cloud_mask_tw = ((d_nrbr >= twbs_th) & (sky_mask > 0)).astype(np.uint8)

    # Fixed-threshold mask
    cloud_mask_f = ((nrbr_orig <= fixed_th) & (sky_mask > 0)).astype(np.uint8)

    # Initial overall cloudiness based on TWBS
    cloudiness_tw = _cloud_fraction(cloud_mask_tw, sky_mask)

    # Default
    sun_cloudiness = 0.0

    # Low-cloudiness regime
    if cloudiness_tw < min_cloudiness_merge:
        cloud_mask_twf = cloud_mask_tw.copy()

    # High-cloudiness regime
    elif cloudiness_tw >= cloudiness_th:
        cloud_mask_twf = cloud_mask_f.copy()

    # Intermediate regime
    else:
        # Merged mask: cloudy if dNRBR >= twbs_th OR NRBR <= fixed_th
        cloud_mask_merge = (
            ((cloud_mask_tw > 0) | (cloud_mask_f > 0)) & (sky_mask > 0)
        ).astype(np.uint8)

        # Measure sun cloudiness on the circumsolar rim
        sun_cloudiness = _cloud_fraction(cloud_mask_tw, circ_sun_rim_mask)

        # Outside circumsolar disk -> merged mask
        outside_mask = (
            ((cloud_mask_merge > 0) & (outside_circ_mask > 0))
        ).astype(np.uint8)

        # Inside circumsolar disk -> choose TWBS or fixed-threshold
        if sun_cloudiness < sun_cloudiness_th:
            inside_mask = ((cloud_mask_tw > 0) & (circ_sun_disk_mask > 0)).astype(np.uint8)
        else:
            inside_mask = ((cloud_mask_f > 0) & (circ_sun_disk_mask > 0)).astype(np.uint8)

        cloud_mask_twf = ((outside_mask > 0) | (inside_mask > 0)).astype(np.uint8)

    cloud_cover = _cloud_fraction(cloud_mask_twf, sky_mask)

    return (
        cloud_cover,
        cloud_mask_tw,
        cloud_mask_twf,
        cloud_mask_f,
        circ_sun_disk_mask,
        circ_sun_rim_mask,
        sun_cloudiness,
    )
