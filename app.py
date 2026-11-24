"""PET Virtual Laboratory: an interactive teaching application.

Educational simulations only. This is not a clinical reconstruction or QC package.
"""
from __future__ import annotations

import base64
import io
import math
from functools import lru_cache
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from scipy.ndimage import gaussian_filter
from skimage.transform import iradon

# -----------------------------------------------------------------------------
# Page setup and visual style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PET Virtual Laboratory", page_icon="☢️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 16px; color: white;
           background: linear-gradient(110deg,#14243b,#24516a 65%,#287b78); margin-bottom: 1rem;}
    .hero h1 {margin:0 0 .35rem 0; font-size:2.0rem;}
    .hero p {margin:0; color:#dceaf2;}
    .small-note {font-size:.85rem; color: #687584;}
    div[data-testid="stMetric"] {background:rgba(120,140,160,.08); padding:.65rem .8rem; border-radius:10px;}
    .stTabs [data-baseweb="tab"] {font-weight:650;}
    </style>
    """,
    unsafe_allow_html=True,
)

ISOTOPES: Dict[str, dict] = {
    # Mean positron ranges in water are from the IAEA handbook. The projected
    # positron-range FWHM values are representative literature values used to
    # parameterise an educational Gaussian-equivalent 2-D kernel. Real kernels
    # are non-Gaussian (cusp with tails), material dependent and ideally obtained
    # from beta-spectrum / Monte Carlo transport for the selected medium.
    "F-18": {"half_life_min": 109.8, "endpoint_mev": 0.634, "mean_range_mm": 0.5, "range_fwhm_mm": 0.54, "beta_plus_branch": 0.9686},
    "C-11": {"half_life_min": 20.3, "endpoint_mev": 0.960, "mean_range_mm": 1.1, "range_fwhm_mm": 0.92, "beta_plus_branch": 0.99750},
    "N-13": {"half_life_min": 9.97, "endpoint_mev": 1.198, "mean_range_mm": 1.3, "range_fwhm_mm": 1.49, "beta_plus_branch": 0.99818},
    "O-15": {"half_life_min": 2.04, "endpoint_mev": 1.732, "mean_range_mm": 2.5, "range_fwhm_mm": 2.48, "beta_plus_branch": 0.99885},
    "Ga-68": {"half_life_min": 67.7, "endpoint_mev": 1.899, "mean_range_mm": 0.8, "range_fwhm_mm": 2.83, "beta_plus_branch": 0.8914},
    "Rb-82": {"half_life_min": 1.27, "endpoint_mev": 3.35, "mean_range_mm": 1.5, "range_fwhm_mm": 6.14, "beta_plus_branch": 0.9543},
}


def make_phantom(kind: str = "NEMA-style hot-sphere phantom", n: int = 64) -> np.ndarray:
    """Create a simple 2-D educational phantom, normalised to a unit background."""
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    r = np.sqrt(x * x + y * y)
    img = np.zeros((n, n), dtype=np.float32)
    body = r <= 0.88
    img[body] = 1.0
    if kind == "Uniform cylinder":
        return img
    if kind == "Two hot sources":
        return make_two_source_phantom(n=n, separation_mm=8.0, diameter_mm=2.5,
                                       hot_activity=4.0, right_hot_activity=8.0,
                                       right_diameter_mm=5.0)
    elif kind == "Shepp-Logan-like anatomy":
        # A compact stylised ellipse phantom, useful for illustrating reconstruction.
        ellipses = [
            (0.0, 0.0, 0.72, 0.88, 1.0),
            (0.0, 0.0, 0.60, 0.76, -0.18),
            (-0.22, 0.0, 0.12, 0.30, 0.28),
            (0.22, 0.0, 0.12, 0.30, 0.28),
            (0.0, 0.32, 0.25, 0.10, 0.20),
            (0.0, -0.32, 0.25, 0.10, 0.20),
        ]
        img[:] = 0
        for cx, cy, rx, ry, val in ellipses:
            img[((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1] += val
        img = np.clip(img, 0, None)
    else:  # NEMA-style sphere challenge
        spheres = [
            (-0.43, -0.30, 0.075, 4.0), (-0.13, -0.32, 0.095, 3.5),
            (0.22, -0.28, 0.12, 3.0), (-0.35, 0.16, 0.14, 2.8),
            (0.10, 0.18, 0.17, 2.5), (0.43, 0.40, 0.10, 3.2),
        ]
        for cx, cy, rad, val in spheres:
            img[(x - cx) ** 2 + (y - cy) ** 2 <= rad**2] = val
    return img


def make_two_source_phantom(n: int = 64, separation_mm: float = 7.0,
                            diameter_mm: float = 2.5, hot_activity: float = 7.0,
                            pixel_mm: float = 1.0,
                            right_hot_activity: float | None = 4.0,
                            right_diameter_mm: float | None = 4.5) -> np.ndarray:
    """Create two non-overlapping hot disks with independent size and activity.

    ``diameter_mm`` and ``hot_activity`` describe the left source. The optional
    ``right_diameter_mm`` and ``right_hot_activity`` describe the right source.
    The surrounding body has unit activity. The default pair is intentionally
    asymmetric (3-mm / 4× on the left; 5-mm / 8× on the right) so size,
    intensity, spill-in and recovery can be compared.
    """
    yy, xx = np.mgrid[:n, :n]
    xx = (xx - (n - 1) / 2.0) * pixel_mm
    yy = ((n - 1) / 2.0 - yy) * pixel_mm
    body_radius_mm = 0.88 * (n * pixel_mm / 2.0)
    img = np.zeros((n, n), dtype=np.float32)
    body = xx**2 + yy**2 <= body_radius_mm**2
    img[body] = 1.0
    left_radius = max(float(diameter_mm), 0.5) / 2.0
    right_diameter = float(diameter_mm if right_diameter_mm is None else right_diameter_mm)
    right_radius = max(right_diameter, 0.5) / 2.0
    half_sep = float(separation_mm) / 2.0
    right_activity = float(hot_activity if right_hot_activity is None else right_hot_activity)
    left_roi = (xx + half_sep)**2 + yy**2 <= left_radius**2
    right_roi = (xx - half_sep)**2 + yy**2 <= right_radius**2
    if np.any(left_roi & right_roi):
        raise ValueError("The two hot-source ROIs overlap. Increase centre separation or reduce source diameters.")
    img[left_roi] = float(hot_activity)
    img[right_roi] = right_activity
    return img


def two_source_masks(n: int = 64, separation_mm: float = 7.0,
                     diameter_mm: float = 2.5, pixel_mm: float = 1.0,
                     right_diameter_mm: float | None = 4.5):
    """Return separate left/right source ROIs, remote background and gap ROI."""
    yy, xx = np.mgrid[:n, :n]
    xx = (xx - (n - 1) / 2.0) * pixel_mm
    yy = ((n - 1) / 2.0 - yy) * pixel_mm
    left_radius = max(float(diameter_mm), 0.5) / 2.0
    right_diameter = float(diameter_mm if right_diameter_mm is None else right_diameter_mm)
    right_radius = max(right_diameter, 0.5) / 2.0
    half_sep = float(separation_mm) / 2.0
    left = (xx + half_sep)**2 + yy**2 <= left_radius**2
    right = (xx - half_sep)**2 + yy**2 <= right_radius**2
    body = xx**2 + yy**2 <= (0.88 * n * pixel_mm / 2.0)**2
    background = body & ~(left | right) & (np.abs(yy) > max(diameter_mm, right_diameter))
    # Put the gap ROI midway between source centres. Its radius is limited by
    # the physical edge-to-edge gap so it does not overlap either source mask.
    edge_gap = max(float(separation_mm) - (float(diameter_mm) + right_diameter)/2.0, 0.0)
    gap_radius = max(0.5*pixel_mm, min(1.0*pixel_mm, edge_gap/3.0))
    gap = body & (xx**2 + yy**2 <= gap_radius**2)
    return left, right, background, gap


def quantify_two_sources(image: np.ndarray, target: np.ndarray,
                         separation_mm: float, diameter_mm: float,
                         hot_activity: float = 7.0, pixel_mm: float = 1.0,
                         right_hot_activity: float | None = 4.0,
                         right_diameter_mm: float | None = 4.5):
    """Calculate source-specific recovery and inter-source cross-talk.

    Each hot ROI follows its own physical source diameter. Contrast recovery
    is background-subtracted and referenced to the same ROI in the physical
    target, avoiding a misleading raw-mean comparison between unequal sources.
    """
    left, right, bg, gap = two_source_masks(image.shape[0], separation_mm,
                                           diameter_mm, pixel_mm, right_diameter_mm)
    bg_mean = float(np.mean(image[bg])) if np.any(bg) else 1.0
    target_bg = float(np.mean(target[bg])) if np.any(bg) else 1.0
    scaled = np.asarray(image, dtype=float) / max(bg_mean, 1e-9)
    target_scaled = np.asarray(target, dtype=float) / max(target_bg, 1e-9)
    left_mean = float(np.mean(scaled[left])) if np.any(left) else float('nan')
    right_mean = float(np.mean(scaled[right])) if np.any(right) else float('nan')
    target_left = float(np.mean(target_scaled[left])) if np.any(left) else float('nan')
    target_right = float(np.mean(target_scaled[right])) if np.any(right) else float('nan')
    gap_mean = float(np.mean(scaled[gap])) if np.any(gap) else float('nan')
    mean_hot = float(np.nanmean([left_mean, right_mean]))
    left_recovery = 100.0 * (left_mean - 1.0) / max(target_left - 1.0, 1e-9)
    right_recovery = 100.0 * (right_mean - 1.0) / max(target_right - 1.0, 1e-9)
    right_nominal_activity = float(hot_activity if right_hot_activity is None else right_hot_activity)
    left_nominal = 100.0 * (left_mean - 1.0) / max(hot_activity - 1.0, 1e-9)
    right_nominal = 100.0 * (right_mean - 1.0) / max(right_nominal_activity - 1.0, 1e-9)
    valley_ratio = 100.0 * (gap_mean - 1.0) / max(mean_hot - 1.0, 1e-9)
    return {
        'image_bg_normalised': scaled,
        'background_mean': bg_mean,
        'left_mean': left_mean, 'right_mean': right_mean,
        'left_recovery_vs_target_pct': left_recovery,
        'right_recovery_vs_target_pct': right_recovery,
        'left_recovery_vs_nominal_pct': left_nominal,
        'right_recovery_vs_nominal_pct': right_nominal,
        'gap_mean': gap_mean, 'valley_to_peak_pct': valley_ratio,
        'left_roi': left, 'right_roi': right, 'background_roi': bg, 'gap_roi': gap,
    }


def fig_two_source_sinogram_paths(separation_mm: float, n_angles: int = 72,
                                  n_detectors: int = 64, detector_fwhm_mm: float = 4.0,
                                  left_activity: float = 4.0, right_activity: float = 8.0,
                                  left_diameter_mm: float = 3.0,
                                  right_diameter_mm: float = 5.0):
    """Plot centre trajectories and projected source-width bands for two unequal sources."""
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    det_positions = np.linspace(-0.70 * 64, 0.70 * 64, n_detectors)
    half = separation_mm / 2.0
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bin_spacing_mm = float(det_positions[1] - det_positions[0]) if n_detectors > 1 else 1.0
    for x_mm, label, color, activity, diameter in [
        (-half, f'Left: {left_diameter_mm:g} mm, {left_activity:g}×', '#36a98b', left_activity, left_diameter_mm),
        (half, f'Right: {right_diameter_mm:g} mm, {right_activity:g}×', '#e08b3e', right_activity, right_diameter_mm),
    ]:
        s = x_mm * np.cos(np.deg2rad(angles))
        bins = np.interp(s, det_positions, np.arange(n_detectors))
        width_bins = max(float(diameter), 0.0) / (2.0 * max(bin_spacing_mm, 1e-9))
        ax.fill_between(angles, bins-width_bins, bins+width_bins, color=color, alpha=.12, linewidth=0)
        ax.plot(angles, bins, lw=2.0, label=label, color=color)
    ax.set_xlabel('Projection angle (degrees)'); ax.set_ylabel('Detector bin (approx.)')
    ax.set_title('Ideal sinogram trajectories (unequal source activities)')
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    ax.text(.02,.03,f'Separation: {separation_mm:.1f} mm · detector FWHM: {detector_fwhm_mm:.1f} mm',
            transform=ax.transAxes,fontsize=8,color='#555')
    fig.tight_layout()
    return fig


def fig_point_projection(source_x: float, source_y: float, angle_deg: float):
    """Show how a selected point projects to one detector coordinate and a sinogram path."""
    x_mm, y_mm = float(source_x) * 25.0, float(source_y) * 25.0
    theta = math.radians(float(angle_deg))
    s_mm = x_mm * math.cos(theta) + y_mm * math.sin(theta)
    angles = np.linspace(0.0, 180.0, 361)
    path = x_mm * np.cos(np.deg2rad(angles)) + y_mm * np.sin(np.deg2rad(angles))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    body = plt.Circle((0, 0), 25.0, color='#a7b6c7', alpha=0.12)
    ax0.add_patch(body)
    ax0.add_patch(plt.Circle((0, 0), 25.0, fill=False, color='#8191a3', lw=1.2))
    ax0.scatter([x_mm], [y_mm], s=75, color='#df8a36', edgecolor='black', linewidth=.5, zorder=4)
    # The LOR normal is n=(cos(theta), sin(theta)); all points on the line
    # satisfy x*cos(theta) + y*sin(theta) = s.
    normal = np.array([math.cos(theta), math.sin(theta)])
    tangent = np.array([-math.sin(theta), math.cos(theta)])
    tt = np.linspace(-26, 26, 100)
    line = s_mm * normal[:, None] + tangent[:, None] * tt[None, :]
    ax0.plot(line[0], line[1], color='#299c85', lw=2, label='Selected projection line')
    ax0.arrow(0, 0, 12*normal[0], 12*normal[1], width=.12, head_width=1.5,
              length_includes_head=True, color='#299c85', alpha=.85)
    ax0.set_xlim(-28, 28); ax0.set_ylim(-28, 28); ax0.set_aspect('equal')
    ax0.set_xlabel('x (mm)'); ax0.set_ylabel('y (mm)')
    ax0.set_title('Object space: selected source and projection')
    ax0.grid(alpha=.18); ax0.legend(fontsize=7, loc='upper right')
    ax1.plot(angles, path, color='#397eb4', lw=2, label='Point-source sinogram path')
    ax1.scatter([angle_deg], [s_mm], color='#df8a36', s=65, zorder=4, label=f'θ={angle_deg:.0f}°')
    ax1.axhline(s_mm, color='#df8a36', ls=':', lw=1)
    ax1.set_xlim(0, 180); ax1.set_xlabel('Projection angle θ (degrees)')
    ax1.set_ylabel('Detector offset s (mm)'); ax1.set_title('Sinogram: point-source trajectory')
    ax1.grid(alpha=.2); ax1.legend(fontsize=7)
    fig.tight_layout()
    return fig, s_mm


@lru_cache(maxsize=12)
def build_projector(n: int, n_angles: int, n_detectors: int) -> sparse.csr_matrix:
    """Build a sparse, sampled parallel-beam projection matrix.

    Each matrix row is one ray and each column is one image pixel. This is a
    simplified educational system model (no detector blur, attenuation or 3-D
    scanner geometry).
    """
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    det_positions = np.linspace(-0.70 * n, 0.70 * n, n_detectors)
    t_values = np.arange(-0.78 * n, 0.78 * n, 0.5)
    rows, cols, vals = [], [], []
    row_id = 0
    for angle in np.deg2rad(angles):
        nx, ny = np.cos(angle), np.sin(angle)
        tx, ty = -ny, nx
        for s in det_positions:
            xx = s * nx + t_values * tx
            yy = s * ny + t_values * ty
            cc = np.rint(xx + (n - 1) / 2).astype(int)
            rr = np.rint((n - 1) / 2 - yy).astype(int)
            good = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
            pix = rr[good] * n + cc[good]
            if pix.size:
                unique, counts = np.unique(pix, return_counts=True)
                rows.extend([row_id] * len(unique))
                cols.extend(unique.tolist())
                vals.extend((counts * 0.5).astype(float).tolist())
            row_id += 1
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n_angles * n_detectors, n * n)).tocsr()
    return mat


@lru_cache(maxsize=24)
def build_sinogram_detector_psf_operator(n_angles: int, n_detectors: int,
                                         sigma_image_px_rounded: float,
                                         image_size: int = 64) -> sparse.csr_matrix:
    """Blur detector-offset bins independently within each projection view.

    Detector/system response broadens the measured detector-pair coordinate.
    In this 2-D parallel-beam teaching model it is represented as a Gaussian
    convolution along the detector-offset axis of each projection, rather than
    as a 2-D blur of the object image. Sigma is supplied in image-pixel units
    and converted to detector-bin units using the projector's bin spacing.
    """
    sigma_image_px = float(sigma_image_px_rounded)
    n_angles, n_detectors, image_size = int(n_angles), int(n_detectors), int(image_size)
    if sigma_image_px <= 1e-6:
        return sparse.identity(n_angles*n_detectors, format="csr", dtype=float)
    detector_bin_spacing_px = (1.4 * image_size) / max(n_detectors - 1, 1)
    sigma_bins = sigma_image_px / max(detector_bin_spacing_px, 1e-9)
    radius = max(1, int(math.ceil(3.0*sigma_bins)))
    rows, cols, vals = [], [], []
    for angle_index in range(n_angles):
        base = angle_index * n_detectors
        for output_bin in range(n_detectors):
            input_bins = np.arange(max(0, output_bin-radius),
                                   min(n_detectors, output_bin+radius+1))
            weights = np.exp(-0.5*((input_bins-output_bin)/sigma_bins)**2)
            weights /= max(float(weights.sum()), 1e-12)
            row = base + output_bin
            rows.extend([row] * len(input_bins))
            cols.extend((base + input_bins).tolist())
            vals.extend(weights.tolist())
    size = n_angles*n_detectors
    return sparse.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()

@lru_cache(maxsize=24)
def build_image_range_operator(n: int, sigma_px_rounded: float) -> sparse.csr_matrix:
    """Sparse 2-D Gaussian-equivalent positron-range response H for the image grid.

    This is an explicit isotope-specific educational response operator. The
    literature range profiles are not truly Gaussian; this compact model matches
    their representative FWHM but does not reproduce the cusp and long tails of
    a full beta-spectrum / material-specific Monte Carlo kernel.
    """
    n = int(n)
    sigma = float(sigma_px_rounded)
    if sigma <= 1e-6:
        return sparse.identity(n*n, format="csr", dtype=float)
    radius = max(1, int(math.ceil(3.0*sigma)))
    offsets = [(dy, dx, math.exp(-0.5*(dx*dx+dy*dy)/(sigma*sigma)))
                for dy in range(-radius, radius+1)
                for dx in range(-radius, radius+1)]
    rows, cols, vals = [], [], []
    for y in range(n):
        for x in range(n):
            row = y*n+x
            local = []
            for dy, dx, weight in offsets:
                yy, xx = y+dy, x+dx
                if 0 <= yy < n and 0 <= xx < n:
                    local.append((yy*n+xx, weight))
            norm = sum(w for _, w in local)
            for col, weight in local:
                rows.append(row); cols.append(col); vals.append(weight/max(norm, 1e-12))
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n*n, n*n)).tocsr()


def gaussian_kernel_1d(fwhm_mm: float, x_mm: np.ndarray) -> np.ndarray:
    """Unit-peak Gaussian-equivalent response profile for teaching plots."""
    sigma = max(float(fwhm_mm), 1e-6) / 2.354820045
    return np.exp(-0.5*(np.asarray(x_mm, dtype=float)/sigma)**2)


def fig_isotope_kernel_comparison(isotope: str, detector_fwhm_mm: float):
    """Show isotope-specific range response separately from scanner detector PSF."""
    range_fwhm = float(ISOTOPES[isotope]["range_fwhm_mm"])
    xx = np.linspace(-10, 10, 801)
    yrange = gaussian_kernel_1d(range_fwhm, xx)
    ydet = gaussian_kernel_1d(detector_fwhm_mm, xx)
    effective = math.sqrt(range_fwhm**2 + float(detector_fwhm_mm)**2)
    yeff = gaussian_kernel_1d(effective, xx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    axes[0].plot(xx, yrange, label=f"{isotope} range kernel · {range_fwhm:.2f} mm FWHM", lw=2)
    axes[0].plot(xx, ydet, label=f"Scanner detector PSF · {detector_fwhm_mm:.2f} mm FWHM", lw=2)
    axes[0].plot(xx, yeff, label=f"Combined effective response · {effective:.2f} mm", lw=2, ls="--")
    axes[0].set_xlim(-7, 7); axes[0].set_xlabel("Offset (mm)"); axes[0].set_ylabel("Relative response")
    axes[0].set_title("Separate physical response components"); axes[0].grid(alpha=.22); axes[0].legend(fontsize=7)
    values = [ISOTOPES[k]["range_fwhm_mm"] for k in ISOTOPES]
    labels = list(ISOTOPES)
    effs = [math.sqrt(float(v)**2 + float(detector_fwhm_mm)**2) for v in values]
    ypos = np.arange(len(labels))
    axes[1].barh(ypos, values, label="Positron-range FWHM")
    axes[1].scatter(effs, ypos, marker="D", color="#b54c39", label="Combined FWHM estimate", zorder=3)
    axes[1].axvline(detector_fwhm_mm, color="#356b9b", ls=":", label="Detector PSF FWHM")
    axes[1].set_yticks(ypos, labels); axes[1].invert_yaxis(); axes[1].set_xlabel("FWHM (mm)")
    axes[1].set_title("Same scanner, isotope-specific range"); axes[1].grid(axis="x", alpha=.22); axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig

def decay_average_activity_factor(half_life_min: float, duration_min: float) -> float:
    """Mean activity during a scan divided by activity at scan start."""
    lam = math.log(2.0) / max(float(half_life_min),1e-9)
    x = lam * max(float(duration_min),0.0)
    return float((1.0-math.exp(-x))/x) if x>1e-9 else 1.0


def isotope_range_sigma_px(isotope: str, pixel_mm: float) -> float:
    """Convert the selected isotope's projected positron-range FWHM to pixels.

    The isotope-specific FWHM values are representative literature values in
    water. The actual range kernel is non-Gaussian and tissue dependent; this
    Gaussian-equivalent approximation is intended for teaching comparisons.
    """
    fwhm_mm = float(ISOTOPES[isotope]["range_fwhm_mm"])
    return fwhm_mm / 2.354820045 / max(float(pixel_mm), 1e-6)


def simulate_sinogram(image: np.ndarray, n_angles: int, n_detectors: int, counts: int,
                      scatter_fraction: float, random_fraction: float, seed: int = 7,
                      range_sigma_px: float = 0.0, detector_psf_sigma_px: float = 0.0,
                      use_psf_model: bool = False, positron_branch: float = 1.0,
                      decay_factor: float = 1.0, acquisition_duration_min: float = 10.0,
                      model_range_response: bool = False):
    n = image.shape[0]
    A = build_projector(n, n_angles, n_detectors)
    # Positron-range blur is applied to the activity distribution before detection.
    H_range = build_image_range_operator(n, round(float(range_sigma_px), 3))
    range_blurred = (H_range @ np.asarray(image,dtype=float).ravel()).reshape(image.shape)
    # Detector/system response broadens detector-offset bins within each view.
    # Data are always generated with G@A. PSF-aware reconstruction uses G@A;
    # PSF-off deliberately assumes ideal detector bins to demonstrate mismatch.
    G_detector = build_sinogram_detector_psf_operator(
        n_angles, n_detectors, round(float(detector_psf_sigma_px), 3), n)
    A_data = (G_detector @ A).tocsr() if detector_psf_sigma_px > 1e-6 else A
    truth = np.asarray(A_data @ range_blurred.ravel()).ravel()
    # Scale expected counts to a user-selected total, then add a simple uniform background.
    scale = max(counts, 1) / max(truth.sum(), 1e-9)
    # Reference counts correspond to a 10-min scan at unit β+ branch and no decay.
    event_yield_factor = (float(np.clip(positron_branch,0.0,1.0))
                          * float(np.clip(decay_factor,0.0,1.0))
                          * max(float(acquisition_duration_min),0.0)/10.0)
    true_counts = truth * scale * event_yield_factor
    scatter = np.full_like(true_counts, true_counts.mean() * scatter_fraction)
    randoms = np.full_like(true_counts, true_counts.mean() * random_fraction)
    rng = np.random.default_rng(seed)
    prompts = rng.poisson(np.maximum(true_counts + scatter + randoms, 0)).astype(float)
    # Reconstruction response can include either or both known components:
    # H_range is isotope-specific; G_detector is scanner-specific and common
    # to radionuclides on the same scanner.
    A_model = A
    if model_range_response and range_sigma_px > 1e-6:
        A_model = (A_model @ H_range).tocsr()
    if use_psf_model and detector_psf_sigma_px > 1e-6:
        A_model = (G_detector @ A_model).tocsr()
    return prompts, true_counts, scatter, randoms, A_model


def reconstruct(prompts: np.ndarray, A: sparse.csr_matrix, image_shape: Tuple[int, int],
                algorithm: str, iterations: int, scatter: np.ndarray, randoms: np.ndarray,
                n_subsets: int = 6, seed: int = 7) -> np.ndarray:
    """Reconstruct data using backprojection, FBP, MLEM or ordered-subsets EM.

    OSEM partitions projection-angle rows into interleaved subsets. One OSEM
    iteration means one pass through all subsets (therefore n_subsets updates).
    This is a simplified 2-D educational implementation, not scanner software.
    """
    n = image_shape[0]
    n_angles = A.shape[0] // n
    n_det = n
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    corrected = np.maximum(prompts - scatter - randoms, 0)
    if algorithm == "Simple backprojection":
        back = (A.T @ corrected).reshape(n, n)
        return back / max(n_angles, 1)
    if algorithm == "Filtered backprojection (FBP)":
        sino = corrected.reshape(n_angles, n_det).T
        try:
            rec = iradon(sino, theta=theta, filter_name="ramp", circle=False,
                         output_size=n, preserve_range=True)
            return np.maximum(rec, 0)
        except Exception:
            back = (A.T @ corrected).reshape(n, n)
            return np.maximum(back, 0) / max(n_angles, 1)

    y = np.maximum(prompts, 0)
    b = np.maximum(scatter + randoms, 0)
    x = np.full(n * n, max(y.sum() / max(float(A.sum()), 1), 1e-6), dtype=float)
    if algorithm == "MLEM":
        subsets = [np.arange(A.shape[0])]
    else:
        # Interleaved angle subsets: all detector bins from every nth view.
        subsets = [np.concatenate([np.arange(a*n_det, (a+1)*n_det)
                                   for a in range(k, n_angles, max(1, n_subsets))])
                   for k in range(min(max(1, n_subsets), n_angles))]
    for _ in range(max(1, iterations)):
        for rows in subsets:
            As = A[rows]
            ys = y[rows]
            bs = b[rows]
            sensitivity = np.asarray(As.sum(axis=0)).ravel()
            prediction = np.asarray(As @ x).ravel() + bs + 1e-9
            ratio = ys / prediction
            update = np.asarray(As.T @ ratio).ravel()
            x *= update / np.maximum(sensitivity, 1e-9)
            x = np.maximum(x, 0)
    return x.reshape(n, n)



def reconstruct_history(prompts: np.ndarray, A: sparse.csr_matrix,
                        image_shape: Tuple[int, int], scatter: np.ndarray,
                        randoms: np.ndarray, n_subsets: int = 1,
                        max_iterations: int = 6):
    """Return MLEM/OSEM snapshots after each complete nominal iteration.

    n_subsets=1 is the MLEM full-data update. For n_subsets>1, each pass
    sequentially updates the estimate using interleaved projection-angle
    subsets. The returned history enables fair comparison by total subset
    updates (= nominal iterations * subset count).
    """
    n = int(image_shape[0])
    n_det = n
    n_angles = A.shape[0] // n_det
    y = np.maximum(np.asarray(prompts, dtype=float), 0.0)
    b = np.maximum(np.asarray(scatter, dtype=float) + np.asarray(randoms, dtype=float), 0.0)
    x = np.full(n*n, max(float(y.sum()) / max(float(A.sum()), 1.0), 1e-6), dtype=float)
    subset_count = min(max(1, int(n_subsets)), n_angles)
    subsets = [np.concatenate([np.arange(a*n_det, (a+1)*n_det)
                               for a in range(k, n_angles, subset_count)])
               for k in range(subset_count)]
    history=[]
    for iteration in range(1, max(1, int(max_iterations))+1):
        for rows in subsets:
            As=A[rows]
            ys=y[rows]
            bs=b[rows]
            sensitivity=np.asarray(As.sum(axis=0)).ravel()
            prediction=np.asarray(As @ x).ravel()+bs+1e-9
            ratio=ys/prediction
            update=np.asarray(As.T @ ratio).ravel()
            x *= update/np.maximum(sensitivity,1e-9)
            x=np.maximum(x,0.0)
        full_prediction=np.maximum(np.asarray(A @ x).ravel()+b,1e-9)
        log_likelihood=float(np.sum(y*np.log(full_prediction)-full_prediction))
        history.append({"iteration":iteration,
                        "subset_updates":iteration*subset_count,
                        "image":x.reshape(image_shape).copy(),
                        "log_likelihood":log_likelihood})
    return history


def reconstruction_metrics(image: np.ndarray, target: np.ndarray):
    """Task-oriented educational image metrics from masks defined on target."""
    image=np.asarray(image,dtype=float)
    target=np.asarray(target,dtype=float)
    body=target>max(float(np.max(target))*0.03, 0.03)
    bg=body & (target>=0.85) & (target<=1.15)
    if np.count_nonzero(bg)<8:
        bg=body & (target<=np.quantile(target[body],0.60)) if np.any(body) else np.zeros_like(body)
    # Hot structures are deliberately separated from background by target truth.
    bg_mean=float(np.mean(target[bg])) if np.any(bg) else 1.0
    hot=body & (target>max(1.35, bg_mean*1.35))
    if np.count_nonzero(hot)<3 and np.any(body) and float(np.max(target[body]))>bg_mean*1.5:
        hot=body & (target>=np.quantile(target[body],0.85))
    bg_vals=image[bg] if np.any(bg) else np.asarray([0.0])
    mean_bg=float(np.mean(bg_vals))
    sd_bg=float(np.std(bg_vals,ddof=1)) if len(bg_vals)>1 else 0.0
    cv=100.0*sd_bg/max(abs(mean_bg),1e-9)
    snr=mean_bg/max(sd_bg,1e-9)
    if np.any(hot):
        hot_mean=float(np.mean(image[hot]))
        target_hot=float(np.mean(target[hot]))
        target_bg=float(np.mean(target[bg])) if np.any(bg) else 1.0
        measured_contrast=(hot_mean-mean_bg)/max(abs(mean_bg),1e-9)
        true_contrast=(target_hot-target_bg)/max(abs(target_bg),1e-9)
        crc=100.0*measured_contrast/max(abs(true_contrast),1e-9)
        cnr=(hot_mean-mean_bg)/max(sd_bg,1e-9)
    else:
        hot_mean=np.nan; crc=np.nan; cnr=np.nan
    rmse=float(np.sqrt(np.mean((image[body]-target[body])**2))/max(float(np.mean(target[body])),1e-9)) if np.any(body) else np.nan
    return {"Background CV (%)":cv,"Background SNR":snr,
            "Hot contrast recovery (%)":crc,"Hot-to-background CNR":cnr,
            "Normalised RMSE":rmse}


def fig_resolution_path():
    """Annotated imaging-chain diagram showing where resolution effects enter."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(13,3.3))
    ax.set_xlim(0,13); ax.set_ylim(0,3.3); ax.axis('off')
    boxes=[
        (0.15,1.48,1.45,0.85,"True activity\ndistribution"),
        (1.95,1.48,1.55,0.85,"Positron\ntransport"),
        (3.85,1.48,1.55,0.85,"Annihilation\nphotons"),
        (5.75,1.48,1.60,0.85,"Detector / system\nresponse"),
        (7.70,1.48,1.55,0.85,"Measured\nprojections"),
        (9.60,1.48,1.55,0.85,"Reconstruction"),
        (11.45,1.48,1.40,0.85,"Image +\nROI"),
    ]
    fills=['#e8eef7','#eee7fb','#e7f2f7','#fff0dc','#e6f3ef','#e9eef9','#f4e8e8']
    for (x,y,w,h,label),fc in zip(boxes,fills):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',
                             facecolor=fc,edgecolor='#526477',linewidth=1.2)
        ax.add_patch(patch); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9,color='#1d2935',weight='bold')
    for i in range(len(boxes)-1):
        x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x1+0.04,1.90),(x2-0.04,1.90),arrowstyle='-|>',mutation_scale=12,color='#536579',lw=1.4))
    # Physical and computational effects are annotated where they enter.
    ax.annotate('Positron range\n(blurs origin before annihilation)',xy=(2.72,2.34),xytext=(2.72,3.10),
                ha='center',va='center',fontsize=8,color='#7046a4',
                arrowprops=dict(arrowstyle='-|>',color='#7046a4',lw=1.5))
    ax.annotate('Acollinearity + detector PSF\n(blurs measured response)',xy=(6.55,2.34),xytext=(6.55,3.10),
                ha='center',va='center',fontsize=8,color='#a56a16',
                arrowprops=dict(arrowstyle='-|>',color='#a56a16',lw=1.5))
    ax.annotate('PSF modelling enters here\n(system response in forward model)',xy=(10.37,2.30),xytext=(10.37,0.78),
                ha='center',va='center',fontsize=8,color='#276f60',
                arrowprops=dict(arrowstyle='-|>',color='#276f60',lw=1.5))
    ax.annotate('PVE: activity spills across ROI boundary\n→ small structures under-recover',xy=(12.15,1.45),xytext=(12.15,0.35),
                ha='center',va='center',fontsize=8,color='#ad4d4d',
                arrowprops=dict(arrowstyle='-|>',color='#ad4d4d',lw=1.5))
    ax.text(0.2,0.22,'TOF is different: timing narrows the probability distribution along an LOR; it does not remove positron-range or detector blur.',
            fontsize=8,color='#34495e',ha='left')
    fig.tight_layout()
    return fig

def build_nu4_volume(voxel_xy_mm: float = 0.5, voxel_z_mm: float = 1.0,
                     counts_per_unit: float = 1800.0, resolution_fwhm_mm: float = 1.0,
                     seed: int = 13, isotope_range_fwhm_mm: float = 0.0,
                     psf_model: bool = False, positron_branch: float = 1.0,
                     decay_factor: float = 1.0, acquisition_duration_min: float = 10.0):
    """Create a synthetic 3-D NEMA NU 4-2008 image-quality phantom.

    Nominal phantom geometry follows the published 50-mm-long, 15-mm-radius
    design with 15-mm uniform, 20-mm rod and 15-mm spillover sections. This is
    a pedagogical digital phantom, not a substitute for the physical standard.
    """
    from scipy.ndimage import gaussian_filter
    xy = np.arange(-16.0, 16.0 + voxel_xy_mm/2, voxel_xy_mm)
    z = np.arange(0.0, 50.0, voxel_z_mm)
    Y, X = np.meshgrid(xy, xy, indexing="ij")
    radial = X**2 + Y**2 <= 15.0**2
    vol = np.zeros((len(z), len(xy), len(xy)), dtype=np.float32)
    z_spill = z < 15.0
    z_rods = (z >= 15.0) & (z < 35.0)
    z_uniform = z >= 35.0
    # Main chamber: uniform active material in spillover and uniform segments.
    vol[z_spill, :, :] = radial.astype(np.float32)
    vol[z_uniform, :, :] = radial.astype(np.float32)
    # Rod section: inactive surrounding block with five parallel active rods.
    rod_diameters = [1, 2, 3, 4, 5]
    rod_angles_deg = [0, 72, 144, 216, 288]
    rod_centres = []
    for diameter, angle in zip(rod_diameters, rod_angles_deg):
        a = np.deg2rad(angle)
        cx, cy = 7.0*np.cos(a), 7.0*np.sin(a)
        rod_centres.append((cx, cy, diameter))
        rod_mask = (X-cx)**2 + (Y-cy)**2 <= (diameter/2.0)**2
        vol[z_rods, :, :] += rod_mask.astype(np.float32)
    # Two non-emitting chambers in spillover section, inner diameter 8 mm.
    cold_centres = [(-7.5, 0.0, "Water"), (7.5, 0.0, "Air")]
    for cx, cy, _ in cold_centres:
        chamber = (X-cx)**2 + (Y-cy)**2 <= 4.0**2
        for zi in np.where(z_spill)[0]:
            vol[zi][chamber] = 0.0
    # Blur and Poisson sampling emulate finite resolution and counting noise.
    detector_sigma = (resolution_fwhm_mm/2.355/voxel_z_mm,
                      resolution_fwhm_mm/2.355/voxel_xy_mm,
                      resolution_fwhm_mm/2.355/voxel_xy_mm)
    range_sigma = (isotope_range_fwhm_mm/2.355/voxel_z_mm,
                   isotope_range_fwhm_mm/2.355/voxel_xy_mm,
                   isotope_range_fwhm_mm/2.355/voxel_xy_mm)
    effective_sigma = tuple(math.sqrt(a*a+b*b) for a,b in zip(detector_sigma,range_sigma))
    blurred = gaussian_filter(vol, sigma=effective_sigma, mode="constant")
    rng = np.random.default_rng(seed)
    exposure_factor = (float(np.clip(positron_branch,0.01,1.0))
                       * float(np.clip(decay_factor,0.01,1.0))
                       * max(float(acquisition_duration_min),0.0)/10.0)
    effective_counts_per_unit = counts_per_unit * exposure_factor
    sampled = rng.poisson(np.maximum(blurred*effective_counts_per_unit, 0)).astype(np.float32)/effective_counts_per_unit
    if psf_model:
        sampled = richardson_lucy_gaussian(sampled, detector_sigma, iterations=4)
    geometry = {"xy": xy, "z": z, "X": X, "Y": Y, "radial": radial,
                "rod_centres": rod_centres, "cold_centres": cold_centres,
                "voxel_xy_mm": voxel_xy_mm, "voxel_z_mm": voxel_z_mm}
    return vol, sampled, geometry


def nu4_roi_metrics(volume: np.ndarray, geometry: dict):
    """Calculate NU4-style image-quality metrics using protocol-shaped VOIs."""
    X, Y, z = geometry["X"], geometry["Y"], geometry["z"]
    xy_mm, z_mm = geometry["voxel_xy_mm"], geometry["voxel_z_mm"]
    # Central 22.5-mm-diameter x 10-mm-long cylindrical VOI in uniform section.
    uniform_xy = X**2 + Y**2 <= (22.5/2)**2
    uniform_z = (z >= 40.0) & (z < 50.0)
    u = volume[np.ix_(uniform_z, np.ones(len(Y), dtype=bool), np.ones(len(X), dtype=bool))]
    u = u[:, uniform_xy]
    uvals = u[np.isfinite(u)]
    umean = float(np.mean(uvals)) if uvals.size else np.nan
    usd = float(np.std(uvals, ddof=1)) if uvals.size > 1 else np.nan
    umin, umax = float(np.min(uvals)), float(np.max(uvals))
    percent_sd = 100*usd/max(abs(umean), 1e-12)
    dvar = 100*(umax-umin)/max(abs(umean), 1e-12)
    # Sum/average central 10-mm rod slices, position a circular ROI of
    # diameter 2 x rod diameter for maximum response near each rod, and
    # extract an axial profile through that ROI centre.
    rod_results = []
    rod_z = np.where((z >= 20.0) & (z < 30.0))[0]
    summed = np.mean(volume[rod_z], axis=0)
    px = float(xy_mm)
    for cx, cy, diameter in geometry["rod_centres"]:
        roi_radius = diameter  # ROI diameter = 2 x physical rod diameter
        best_mean = -np.inf
        best_yx = (int(np.argmin(abs(geometry["xy"]-cy))), int(np.argmin(abs(geometry["xy"]-cx))))
        for dy in np.arange(-1.0, 1.001, px):
            for dx in np.arange(-1.0, 1.001, px):
                qx, qy = cx+dx, cy+dy
                roi = (X-qx)**2 + (Y-qy)**2 <= roi_radius**2
                if np.any(roi):
                    roi_mean = float(np.mean(summed[roi]))
                    if roi_mean > best_mean:
                        best_mean = roi_mean
                        best_yx = (int(np.argmin(abs(geometry["xy"]-qy))), int(np.argmin(abs(geometry["xy"]-qx))))
        iy, ix = best_yx
        profile = volume[rod_z, iy, ix]
        profile_mean = float(np.mean(profile))
        rc = 100*profile_mean/max(abs(umean), 1e-12)
        profile_cv = 100*float(np.std(profile, ddof=1))/max(abs(profile_mean),1e-12) if len(profile)>1 else 0.0
        rod_results.append({"Rod diameter (mm)": diameter, "Recovery coefficient (%)": rc,
                            "Axial-profile CV (%)": profile_cv, "ROI diameter (mm)": 2*diameter})
    # Central 4-mm-diameter x 7.5-mm-long VOIs in water and air chambers.
    spill_z = (z >= 3.75) & (z < 11.25)
    spill_results = []
    for cx, cy, label in geometry["cold_centres"]:
        roi = (X-cx)**2 + (Y-cy)**2 <= 2.0**2
        vv = volume[np.ix_(spill_z, np.ones(len(Y), dtype=bool), np.ones(len(X), dtype=bool))][:, roi]
        vals = vv.ravel()
        sor = 100*float(np.mean(vals))/max(abs(umean), 1e-12)
        spill_results.append({"Compartment": label, "Spillover ratio (%)": sor,
                              "ROI mean (relative units)": float(np.mean(vals)),
                              "ROI SD (relative units)": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0})
    return {"uniform_mean": umean, "uniform_sd": usd, "uniform_percent_sd": percent_sd,
            "uniform_dvar": dvar, "uniform_min": umin, "uniform_max": umax,
            "rods": pd.DataFrame(rod_results), "spill": pd.DataFrame(spill_results),
            "uniform_roi_xy": uniform_xy, "uniform_roi_z": uniform_z}


def build_acr_pet_slice(pixel_mm: float = 1.0, hot_to_bg: float = 2.5,
                        resolution_fwhm_mm: float = 5.0, noise_pct: float = 2.0,
                        seed: int = 29, isotope_range_fwhm_mm: float = 0.0,
                        psf_model: bool = False, positron_branch: float = 1.0,
                        decay_factor: float = 1.0, acquisition_duration_min: float = 10.0):
    """Synthetic ACR PET faceplate slice for teaching ROI placement/metrics.

    The phantom outline and component sizes are based on the ACR PET phantom
    description. Component positions are schematic, not a metrology drawing.
    """
    from scipy.ndimage import gaussian_filter
    coord = np.arange(-110.0, 110.0+pixel_mm/2, pixel_mm)
    Y, X = np.meshgrid(coord, coord, indexing="ij")
    body = X**2+Y**2 <= 108.0**2
    img = np.zeros_like(X, dtype=np.float32)
    img[body] = 1.0
    # Schematic positions on a ring to keep the central 65-mm background ROI clear.
    hot_specs = [(75, 0, 25, "25 mm hot"), (0, 75, 16, "16 mm hot"),
                 (-75, 0, 12, "12 mm hot"), (0, -75, 8, "8 mm hot")]
    cold_specs = [(53, 53, "Air"), (-53, 53, "Water"), (-53, -53, "Teflon")]
    for cx, cy, d, _ in hot_specs:
        img[(X-cx)**2+(Y-cy)**2 <= (d/2)**2] = hot_to_bg
    for cx, cy, _ in cold_specs:
        img[(X-cx)**2+(Y-cy)**2 <= (12.5)**2] = 0.0
    detector_sigma = resolution_fwhm_mm/2.355/pixel_mm
    range_sigma = isotope_range_fwhm_mm/2.355/pixel_mm
    effective_sigma = math.sqrt(detector_sigma**2+range_sigma**2)
    blurred = gaussian_filter(img, sigma=effective_sigma, mode="constant")
    rng = np.random.default_rng(seed)
    exposure_factor = float(positron_branch)*float(decay_factor)*max(float(acquisition_duration_min),0.0)/10.0
    branch_noise_pct = noise_pct / math.sqrt(max(exposure_factor,0.01))
    sampled = np.maximum(blurred + rng.normal(0, branch_noise_pct/100.0, blurred.shape), 0)
    if psf_model:
        sampled = richardson_lucy_gaussian(sampled, detector_sigma, iterations=4)
    geometry = {"X": X, "Y": Y, "coord": coord, "body": body,
                "hot_specs": hot_specs, "cold_specs": cold_specs, "pixel_mm": pixel_mm}
    return img, sampled.astype(np.float32), geometry


def acr_roi_metrics(image: np.ndarray, geometry: dict):
    """Apply ACR's central 60-70 mm background and common 25-mm ROI approach."""
    X, Y = geometry["X"], geometry["Y"]
    bg_roi = X**2+Y**2 <= (32.5)**2  # 65-mm diameter, centered, avoid cylinders
    bg_mean = float(np.mean(image[bg_roi]))
    bg_sd = float(np.std(image[bg_roi], ddof=1))
    rows=[]
    hot_max={}
    for cx,cy,d,label in geometry["hot_specs"]:
        roi=(X-cx)**2+(Y-cy)**2 <= (12.5)**2  # same 25-mm diameter ROI for all hot cylinders
        vals=image[roi]
        vmax=float(np.max(vals))
        rows.append({"Component":label,"ROI diameter (mm)":25,"Mean SUV":float(np.mean(vals)),
                     "Maximum SUV":vmax,"Minimum SUV":float(np.min(vals))})
        hot_max[d]=vmax
    for cx,cy,label in geometry["cold_specs"]:
        roi=(X-cx)**2+(Y-cy)**2 <= (12.5)**2
        vals=image[roi]
        rows.append({"Component":label,"ROI diameter (mm)":25,"Mean SUV":float(np.mean(vals)),
                     "Maximum SUV":float(np.max(vals)),"Minimum SUV":float(np.min(vals))})
    max25=hot_max[25]
    ratio16=hot_max[16]/max(max25,1e-12)
    criteria={"Mean background SUV 0.90–1.10":0.90 <= bg_mean <= 1.10,
              "Maximum SUV, 25-mm cylinder >1.87 and <2.91":1.87 < max25 < 2.91,
              "Maximum SUV 16-mm / 25-mm ratio >0.70":ratio16 > 0.70}
    return {"background_mean":bg_mean,"background_sd":bg_sd,"max25":max25,
            "ratio16_25":ratio16,"table":pd.DataFrame(rows),"criteria":criteria,
            "background_roi":bg_roi}



def build_nu2_iec_slice_stack(pixel_mm: float = 1.0, sphere_background_ratio: float = 4.0,
                              resolution_fwhm_mm: float = 5.0, noise_pct: float = 2.0,
                              seed: int = 41, isotope_range_fwhm_mm: float = 0.0,
                              psf_model: bool = False, positron_branch: float = 1.0,
                              decay_factor: float = 1.0, acquisition_duration_min: float = 10.0):
    """Synthetic central 5-slice NEMA/IEC NU 2 image-quality phantom demo.

    Sphere diameters, four-hot/two-cold configuration and low-density lung
    insert are represented. Sphere/insert placement and the teaching ROI grid
    are schematic, not a dimensional drawing or complete NU 2 implementation. PSF restoration is a Gaussian-kernel teaching approximation.
    """
    from scipy.ndimage import gaussian_filter
    x=np.arange(-150,150+pixel_mm/2,pixel_mm)
    y=np.arange(-115,115+pixel_mm/2,pixel_mm)
    Y,X=np.meshgrid(y,x,indexing="ij")
    body=(X/135.0)**2+(Y/100.0)**2 <= 1.0
    sphere_ds=[10,13,17,22,28,37]
    angles=np.deg2rad([-90,-30,30,90,150,210])
    centres=[(62*np.cos(a),62*np.sin(a)) for a in angles]
    zvals=np.array([-20.,-10.,0.,10.,20.])
    stack=[]
    for iz,z in enumerate(zvals):
        im=np.zeros_like(X,dtype=float); im[body]=1.0
        im[X**2+Y**2 <= 25.0**2]=0.30  # low-density lung insert teaching signal
        for idx,(d,(cx,cy)) in enumerate(zip(sphere_ds,centres)):
            r2=(d/2)**2-z**2
            if r2<=0: continue
            sphere=(X-cx)**2+(Y-cy)**2 <= r2
            im[sphere]=sphere_background_ratio if idx<4 else 0.0
        detector_sigma=resolution_fwhm_mm/2.355/pixel_mm
        range_sigma=isotope_range_fwhm_mm/2.355/pixel_mm
        effective_sigma=math.sqrt(detector_sigma**2+range_sigma**2)
        blurred=gaussian_filter(im,sigma=effective_sigma,mode="constant")
        rng=np.random.default_rng(seed+iz)
        exposure_factor=float(positron_branch)*float(decay_factor)*max(float(acquisition_duration_min),0.0)/10.0
        branch_noise_pct=noise_pct/math.sqrt(max(exposure_factor,0.01))
        sampled=np.maximum(blurred+rng.normal(0,branch_noise_pct/100.0,blurred.shape),0)
        if psf_model:
            sampled=richardson_lucy_gaussian(sampled,detector_sigma,iterations=4)
        stack.append(sampled)
    # Select reproducible background ROI centres for each diameter. Candidate
    # circles must remain inside the torso, avoid the lung insert, and retain
    # at least 15 mm clearance from the sphere surfaces. This is a teaching
    # placement algorithm; the licensed NU 2 standard governs formal ROIs.
    bg_centres_by_d={}
    for d in sphere_ds:
        radius=d/2.0
        candidates=[]
        for by in np.arange(-85,86,5):
            for bx in np.arange(-125,126,5):
                # Sample the ROI circumference/interior to ensure it stays in body.
                rr=np.arange(0,radius+0.1,2.5)
                aa=np.arange(0,2*np.pi, np.deg2rad(15))
                ox=(rr[:,None]*np.cos(aa)[None,:]).ravel()
                oy=(rr[:,None]*np.sin(aa)[None,:]).ravel()
                ix=np.clip(np.rint((by+oy-y[0])/pixel_mm).astype(int),0,len(y)-1)
                jx=np.clip(np.rint((bx+ox-x[0])/pixel_mm).astype(int),0,len(x)-1)
                if not np.all(body[ix,jx]):
                    continue
                if np.hypot(bx,by) < 25.0+radius+2.0:
                    continue
                if any(np.hypot(bx-cx,by-cy) < radius+sd/2+15.0 for sd,(cx,cy) in zip(sphere_ds,centres)):
                    continue
                candidates.append((float(bx),float(by)))
        chosen=[]
        # Greedy farthest-point sampling with non-overlapping ROI circles.
        if candidates:
            chosen.append(max(candidates,key=lambda q:q[0]**2+q[1]**2))
            while len(chosen)<12:
                eligible=[q for q in candidates if all(np.hypot(q[0]-c[0],q[1]-c[1]) >= d+2 for c in chosen)]
                if not eligible: break
                chosen.append(max(eligible,key=lambda q:min(np.hypot(q[0]-c[0],q[1]-c[1]) for c in chosen)))
        bg_centres_by_d[d]=chosen
    geom={"X":X,"Y":Y,"body":body,"sphere_ds":sphere_ds,"centres":centres,
          "zvals":zvals,"bg_centres_by_d":bg_centres_by_d,"pixel_mm":pixel_mm,
          "activity_ratio":sphere_background_ratio}
    return np.stack(stack).astype(np.float32),geom


def nu2_iec_metrics(stack: np.ndarray, geometry: dict):
    """Educational NU 2-style hot-sphere CRC and background-variability metrics."""
    X,Y=geometry["X"],geometry["Y"]
    centre_slice=stack[2]
    sphere_rows=[]; bg_rows=[]
    for idx,(d,(cx,cy)) in enumerate(zip(geometry["sphere_ds"],geometry["centres"])):
        roi=(X-cx)**2+(Y-cy)**2 <= (d/2)**2
        measured=float(np.mean(centre_slice[roi]))
        bg_samples=[]
        for sl in stack:
            for bx,by in geometry["bg_centres_by_d"][d]:
                broi=(X-bx)**2+(Y-by)**2 <= (d/2)**2
                vals=sl[broi]
                if vals.size: bg_samples.append(float(np.mean(vals)))
        bg_samples=np.asarray(bg_samples)
        bg_mean=float(np.mean(bg_samples)) if bg_samples.size else float("nan")
        bg_sd=float(np.std(bg_samples,ddof=1)) if bg_samples.size>1 else 0.0
        if idx<4:
            crc=100*((measured/max(bg_mean,1e-12))-1.0)/(geometry["activity_ratio"]-1.0)
            sphere_rows.append({"Sphere diameter (mm)":d,"Type":"Hot","Mean ROI value":measured,
                                "Local background mean":bg_mean,"Contrast recovery (%)":crc})
        else:
            sphere_rows.append({"Sphere diameter (mm)":d,"Type":"Cold","Mean ROI value":measured,
                                "Local background mean":bg_mean,"Contrast recovery (%)":np.nan})
        bg_rows.append({"ROI diameter (mm)":d,"Mean background value":bg_mean,
                        "Background variability (%)":100*bg_sd/max(abs(bg_mean),1e-12),
                        "Number of background ROIs":len(bg_samples)})
    lung_roi=(X**2+Y**2 <= 20.0**2)
    lung_mean=float(np.mean(centre_slice[lung_roi]))
    lung_error=100*(lung_mean-0.30)/0.30
    return pd.DataFrame(sphere_rows),pd.DataFrame(bg_rows),lung_mean,lung_error

def build_uniform_cylinder(pixel_mm: float = 0.5, diameter_mm: float = 60.0,
                           concentration: float = 1.0, noise_pct: float = 2.0,
                           resolution_fwhm_mm: float = 1.0, seed: int = 31,
                           isotope_range_fwhm_mm: float = 0.0, psf_model: bool = False,
                           positron_branch: float = 1.0, decay_factor: float = 1.0,
                           acquisition_duration_min: float = 10.0):
    from scipy.ndimage import gaussian_filter
    coord=np.arange(-diameter_mm/2-5,diameter_mm/2+5+pixel_mm/2,pixel_mm)
    Y,X=np.meshgrid(coord,coord,indexing="ij")
    mask=X**2+Y**2 <= (diameter_mm/2)**2
    ideal=np.zeros_like(X,dtype=float); ideal[mask]=concentration
    detector_sigma=resolution_fwhm_mm/2.355/pixel_mm
    range_sigma=isotope_range_fwhm_mm/2.355/pixel_mm
    effective_sigma=math.sqrt(detector_sigma**2+range_sigma**2)
    smooth=gaussian_filter(ideal,effective_sigma)
    rng=np.random.default_rng(seed)
    exposure_factor=float(positron_branch)*float(decay_factor)*max(float(acquisition_duration_min),0.0)/10.0
    branch_noise_pct=noise_pct/math.sqrt(max(exposure_factor,0.01))
    sample=np.maximum(smooth+rng.normal(0,branch_noise_pct/100*concentration,smooth.shape),0)
    if psf_model:
        sample=richardson_lucy_gaussian(sample,detector_sigma,iterations=4)
    return ideal,sample,{"X":X,"Y":Y,"mask":mask,"pixel_mm":pixel_mm}

def fig_image(image, title, cmap="magma", vmin=None, vmax=None, colorbar=True):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return fig


def fig_sinogram(sino, title="Sinogram"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(sino, aspect="auto", origin="lower", cmap="viridis",
                   extent=(0, 180, -0.5, sino.shape[0] - 0.5))
    ax.set_xlabel("Projection angle (degrees)"); ax.set_ylabel("Detector bin")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Counts (a.u.)")
    fig.tight_layout()
    return fig


def canvas_animation(isotope: str, mean_range_mm: float) -> str:
    """Small self-contained Canvas animation for continuous particle motion."""
    safe_iso = isotope.replace("'", "")
    return f"""
    <div style='border:1px solid #ccd5df;border-radius:12px;overflow:hidden;background:#0d1722'>
      <canvas id='pet-anim' style='width:100%;height:290px;display:block'></canvas>
      <div style='padding:7px 12px;color:#dbe7f2;font:12px sans-serif'>
        {safe_iso} · illustrative positron track · representative mean range: {mean_range_mm:.1f} mm
      </div>
    </div>
    <script>
    const canvas=document.getElementById('pet-anim'), ctx=canvas.getContext('2d');
    let t0=performance.now();
    function frame(t){{
      const w=canvas.clientWidth,h=290,d=window.devicePixelRatio||1;
      if(canvas.width!==w*d||canvas.height!==h*d){{canvas.width=w*d;canvas.height=h*d;}}
      ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);
      ctx.fillStyle='#0d1722';ctx.fillRect(0,0,w,h);
      const cx=w*.23,cy=h*.50,scale=Math.min(w*.19,45),phase=((t-t0)%6500)/6500;
      ctx.strokeStyle='#7d91a5';ctx.lineWidth=1;ctx.setLineDash([4,5]);
      ctx.beginPath();ctx.ellipse(cx,cy,55,48,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='#a78bdf';ctx.beginPath();ctx.arc(cx,cy,10,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#fff';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText('β⁺ decay',cx,cy+70);
      const endx=cx+scale*(1+phase*2.0), endy=cy-20*Math.sin(phase*Math.PI*2);
      ctx.strokeStyle='#efa04a';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx+9,cy);ctx.quadraticCurveTo(cx+scale*.55,cy-28,endx,endy);ctx.stroke();
      ctx.fillStyle='#efa04a';ctx.beginPath();ctx.arc(endx,endy,7,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#fff';ctx.fillText('e⁺',endx,endy-15);
      if(phase>.65){{
        const p=(phase-.65)/.35, ex=endx,ey=endy;
        ctx.fillStyle='#64b5e8';ctx.beginPath();ctx.arc(ex+10,ey+5,5,0,Math.PI*2);ctx.fill();
        ctx.strokeStyle='#70d6bb';ctx.lineWidth=2.5;
        for(const s of [-1,1]){{ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex+s*100*p,ey+s*65*p);ctx.stroke();}}
        ctx.fillStyle='#dce8f2';ctx.font='11px sans-serif';ctx.fillText('511 keV',ex+65*p,ey-52*p);
        ctx.fillText('511 keV',ex-65*p,ey+60*p);
      }}
      requestAnimationFrame(frame);
    }}requestAnimationFrame(frame);
    </script>
    """



def positronium_animation(isotope: str, branch: str, ps_fraction: int) -> str:
    """Self-contained animated teaching schematic of slowing, Ps formation and annihilation."""
    safe_iso = isotope.replace("'", "")
    safe_branch = branch.replace("'", "")
    html = r"""
    <div style="border:1px solid #34475b;border-radius:14px;overflow:hidden;background:#0c1723;color:#eaf1f7">
      <canvas id="ps-story" style="width:100%;height:330px;display:block"></canvas>
      <div style="display:flex;gap:14px;flex-wrap:wrap;padding:7px 12px;background:#142537;font:12px sans-serif;color:#dbe7f2">
        <span>🟣 nucleus / decay</span><span>🟠 positron</span><span>🔵 electron</span><span>🟢 photons</span><span>🟡 positronium</span>
      </div>
      <div style="padding:8px 12px;color:#c4d1de;font:12px sans-serif" id="ps-caption"></div>
    </div>
    <script>
    (()=>{
      const canvas=document.getElementById('ps-story'); if(!canvas) return;
      const ctx=canvas.getContext('2d'), caption=document.getElementById('ps-caption');
      const branch='__BRANCH__', iso='__ISO__', psFraction=__PSFRAC__, meanRange=__RANGE__;
      let start=performance.now();
      const labels={
        'Direct annihilation':'Direct annihilation: a thermalised positron annihilates with an electron.',
        'Para-positronium (p-Ps)':'p-Ps (singlet, S=0): self-annihilation predominantly produces two photons.',
        'Ortho-Ps in tissue (pick-off)':'o-Ps in tissue: pick-off with an external molecular electron shortens the lifetime and produces two photons.',
        'Ortho-Ps in vacuum (3γ)':'o-Ps in vacuum: the triplet state decays predominantly by three-photon annihilation.'
      };
      function line(x1,y1,x2,y2,color,width=2,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([])}
      function photon(x,y,angle,len,progress,color='#64d5ae'){
        const ex=x+Math.cos(angle)*len*progress,ey=y+Math.sin(angle)*len*progress;
        line(x,y,ex,ey,color,2.6);ctx.fillStyle=color;ctx.beginPath();ctx.arc(ex,ey,4,0,Math.PI*2);ctx.fill();
      }
      function frame(now){
        const w=canvas.clientWidth||700,h=330,d=window.devicePixelRatio||1;
        if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d)}
        ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#0c1723';ctx.fillRect(0,0,w,h);
        const phase=((now-start)%10000)/10000, cx=w*.16,cy=h*.48, speed=Math.min(1,Math.max(0,(phase-.10)/.48));
        ctx.fillStyle='#a78be0';ctx.beginPath();ctx.arc(cx,cy,14,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#f0eafa';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText('proton-rich nucleus',cx,cy+39);
        ctx.fillStyle='#d7e1eb';ctx.font='11px sans-serif';ctx.textAlign='left';ctx.fillText(iso+' β⁺ decay',cx-18,cy-33);
        // Stochastic-looking, fixed curved path with increasingly short steps as the positron slows.
        const trackScale=Math.min(1.8,0.58+meanRange/3.0);
        const pts=[];for(let i=0;i<=12;i++){const q=i/12;pts.push([cx+25+q*w*.30*trackScale,cy-35*Math.sin(q*5.7)*(1-.65*q)+12*Math.sin(q*17)*(1-.8*q)])}
        const end=Math.max(1,Math.floor(speed*12)), frac=speed*12-end;
        ctx.strokeStyle='#efa04a';ctx.lineWidth=2.7;ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<=end;i++)ctx.lineTo(pts[i][0],pts[i][1]);if(end<12)ctx.lineTo(pts[end][0]+(pts[end+1][0]-pts[end][0])*frac,pts[end][1]+(pts[end+1][1]-pts[end][1])*frac);ctx.stroke();
        const ep=pts[Math.min(end,12)];ctx.fillStyle='#efa04a';ctx.beginPath();ctx.arc(ep[0],ep[1],6,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#e6edf5';ctx.fillText('e⁺',ep[0],ep[1]-13);
        // Collision/excitation markers illustrate stochastic energy loss, not individual resolved interactions.
        for(let i=0;i<5;i++){const q=.18+i*.13;const x=cx+25+q*w*.30*trackScale,y=cy-35*Math.sin(q*5.7)*(1-.65*q)+12*Math.sin(q*17)*(1-.8*q);ctx.strokeStyle='#8ba2b8';ctx.globalAlpha=phase>.13+i*.025?.85:.18;ctx.beginPath();ctx.arc(x,y,4+i*.35,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;}
        const thermal=phase>.58, bound=phase>.67, emission=phase>.76;
        ctx.fillStyle=thermal?'#75c9ed':'#9aabba';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText(thermal?'Thermalised / near-thermal e⁺':'Inelastic + elastic interactions',w*.47,h*.84);
        const ax=w*.67, ay=h*.48;
        if(thermal){ctx.fillStyle='#efa04a';ctx.beginPath();ctx.arc(ax-14,ay,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#72b9e8';ctx.beginPath();ctx.arc(ax+14,ay,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#e8f0f7';ctx.fillText('e⁺',ax-14,ay-15);ctx.fillText('e⁻',ax+14,ay-15);ctx.fillText('Direct annihilation or Ps formation',ax,ay+43);}
        let originX=ax,originY=ay;
        if(bound && branch!=='Direct annihilation'){ctx.strokeStyle='#e8c65a';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(ax,ay,33,20,0,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#e8c65a';ctx.font='12px sans-serif';ctx.fillText(branch.includes('Para')?'p-Ps singlet':'o-Ps triplet',ax,ay-31);ctx.fillText(branch.includes('Para')?'antiparallel spins':'parallel spins',ax,ay+42);}
        if(emission){
          if(branch==='Ortho-Ps in vacuum (3γ)'){
            photon(originX,originY,-2.65,Math.min(w*.23,145),Math.min(1,(phase-.76)/.22));photon(originX,originY,-.1,Math.min(w*.23,145),Math.min(1,(phase-.76)/.22));photon(originX,originY,2.0,Math.min(w*.23,145),Math.min(1,(phase-.76)/.22));
            ctx.fillStyle='#74d7b4';ctx.fillText('3γ',originX,originY+68);
          }else{
            if(branch==='Ortho-Ps in tissue (pick-off)'){
              const q=Math.min(1,(phase-.76)/.12);ctx.fillStyle='#72b9e8';ctx.beginPath();ctx.arc(originX+42*q,originY+8,6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#e6edf5';ctx.fillText('external e⁻ (pick-off)',originX+43,originY+28);
            }
            photon(originX,originY,Math.PI,Math.min(w*.23,145),Math.min(1,(phase-.76)/.22));photon(originX,originY,0,Math.min(w*.23,145),Math.min(1,(phase-.76)/.22));ctx.fillStyle='#74d7b4';ctx.fillText('2γ · 511 keV each (approx.)',originX,originY+68);
          }
        }
        // Stage guide
        const stages=['β⁺ emission','Slowing / thermalisation','Direct e⁺e⁻ or Ps formation','Annihilation photons'];
        const active=phase<.10?0:phase<.58?1:phase<.76?2:3;
        const baseY=24;for(let i=0;i<4;i++){let x=16+i*(w-32)/4;ctx.fillStyle=i===active?'#f3ca68':'#71869a';ctx.beginPath();ctx.arc(x+12,baseY,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#d3dfeb';ctx.font='10px sans-serif';ctx.textAlign='left';ctx.fillText(stages[i],x+22,baseY+4);}
        caption.textContent=labels[branch]+' Illustrative Ps formation fraction setting: '+psFraction+'%. Schematic timing is compressed; the animation is not to physical time scale.';
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    })();
    </script>
    """
    range_value = float(ISOTOPES[isotope]["mean_range_mm"])
    return (html.replace('__BRANCH__', safe_branch).replace('__ISO__', safe_iso)
            .replace('__PSFRAC__', str(int(ps_fraction))).replace('__RANGE__', str(range_value)))

def detector_3d_figure(event_type: str):
    """Clear, compact top-down PET coincidence event teaching diagram."""
    fig, ax = plt.subplots(figsize=(4.2, 3.65))
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#4c6f8a", lw=13, alpha=.18))
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    ax.scatter(np.cos(angles), np.sin(angles), s=34, marker="s", c="#3478a8",
               edgecolors="white", linewidths=.45, zorder=5, label="Detector blocks")

    def path(points, color, label):
        pts=np.asarray(points,dtype=float)
        ax.plot(pts[:,0],pts[:,1],color=color,lw=2.8,solid_capstyle="round",zorder=3,label=label)
        for a,b in zip(pts[:-1],pts[1:]):
            d=b-a; norm=np.linalg.norm(d)
            if norm:
                mid=a+.70*d
                ax.annotate("",xy=mid+.08*d/norm,xytext=mid,
                            arrowprops=dict(arrowstyle="-|>",color=color,lw=1.5),zorder=4)

    if event_type == "True":
        o=np.array([0.,0.]); l=np.array([-.94,.34]); r=np.array([.94,-.34])
        ax.scatter(*o,s=75,c="#f2a541",edgecolors="white",zorder=6,label="One annihilation")
        path([o,l],"#15966d","Photon 1"); path([o,r],"#15966d","Photon 2")
        ax.plot([l[0],r[0]],[l[1],r[1]],"--",color="#15966d",lw=1.2,label="True LOR")
        title="TRUE · one annihilation, two detected photons"
    elif event_type == "Scattered":
        o=np.array([0.,0.]); q=np.array([.24,.16]); l=np.array([-.94,.34]); r=np.array([.94,-.34])
        ax.scatter(*o,s=75,c="#f2a541",edgecolors="white",zorder=6,label="One annihilation")
        path([o,l],"#15966d","Unscattered photon"); path([o,q,r],"#d47b27","Scattered photon")
        ax.scatter(*q,marker="x",s=90,lw=2.5,c="#d47b27",zorder=7,label="Scatter interaction")
        ax.plot([l[0],r[0]],[l[1],r[1]],"--",color="#bd4a54",lw=1.8,label="Incorrect apparent LOR")
        title="SCATTER · one photon changes direction"
    elif event_type == "Random":
        o1=np.array([-.28,.18]); o2=np.array([.30,-.20]); d1=np.array([-.94,.34]); d2=np.array([.94,.34])
        ax.scatter(*o1,s=75,c="#f2a541",edgecolors="white",zorder=6,label="Annihilation A")
        ax.scatter(*o2,s=75,c="#a77bd1",edgecolors="white",zorder=6,label="Annihilation B")
        path([o1,d1],"#15966d","Photon from A"); path([o2,d2],"#a77bd1","Photon from B")
        ax.plot([d1[0],d2[0]],[d1[1],d2[1]],"--",color="#bd4a54",lw=1.8,label="False paired LOR")
        title="RANDOM · photons from unrelated annihilations"
    else:
        o=np.array([0.,0.]); d=np.array([-.94,.34])
        ax.scatter(*o,s=75,c="#f2a541",edgecolors="white",zorder=6,label="One annihilation")
        path([o,d],"#718096","Detected photon")
        ax.text(.18,-.18,"Partner photon\nnot detected",color="#bd4a54",fontsize=8,ha="center",va="center",
                bbox=dict(boxstyle="round,pad=.3",fc="white",ec="#bd4a54",alpha=.9))
        title="SINGLE · only one photon is detected"
    ax.set_xlim(-1.28,1.28); ax.set_ylim(-1.25,1.25)
    ax.set_title(title,fontsize=8.5,weight="bold",pad=7); ax.axis("off")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.08),ncol=2,fontsize=7,frameon=False,handlelength=1.7)
    fig.tight_layout(pad=.5)
    return fig


def richardson_lucy_gaussian(image: np.ndarray, sigma, iterations: int = 4) -> np.ndarray:
    """Educational Richardson-Lucy restoration for a known symmetric Gaussian PSF.

    This is used only in the QC/count-rate phantom demonstrations. The main
    MLEM/OSEM laboratory instead includes the PSF directly in its forward model.
    """
    observed = np.clip(np.asarray(image, dtype=float), 0.0, None)
    eps = max(float(observed.max()) * 1e-8, 1e-12)
    estimate = np.maximum(observed.copy(), eps)
    sigma_tuple = sigma if isinstance(sigma, tuple) else float(sigma)
    for _ in range(max(1, int(iterations))):
        predicted = gaussian_filter(estimate, sigma=sigma_tuple, mode="reflect")
        ratio = observed / np.maximum(predicted, eps)
        estimate *= gaussian_filter(ratio, sigma=sigma_tuple, mode="reflect")
        estimate = np.maximum(estimate, 0.0)
    return estimate.astype(np.float32)


def simulate_count_rate_image(activity: float, scatter_pct: float, random_pct: float,
                              apply_scatter: bool, apply_random: bool, seed: int = 17,
                              isotope: str = "F-18", detector_fwhm_mm: float = 4.0,
                              psf_model: bool = False, acquisition_duration_min: float = 10.0):
    """Create a teaching image whose count statistics and background depend on count-rate controls."""
    base_activity = make_phantom("NEMA-style hot-sphere phantom", 64).astype(float)
    body = base_activity > 0
    # Define the background VOI from the known unblurred phantom, not an exact
    # equality test on a blurred image (which can create an empty mask).
    bg = body & (base_activity == 1.0)
    truth = base_activity.copy()
    # Isotope-specific range and detector response both affect apparent lesion recovery.
    range_sigma = isotope_range_sigma_px(isotope, pixel_mm=1.0)
    detector_sigma = detector_fwhm_mm/2.355
    truth = gaussian_filter(truth, sigma=range_sigma) if range_sigma>0 else truth
    # The measured image always contains the physical detector blur. PSF modelling
    # is an image-estimation step, not a change to the scanner's physical resolution.
    truth = gaussian_filter(truth, sigma=detector_sigma)
    # Count-rate curve has a realistic qualitative roll-off; values are deliberately arbitrary.
    beta_plus_branch = float(ISOTOPES[isotope]["beta_plus_branch"])
    decay_factor = decay_average_activity_factor(ISOTOPES[isotope]["half_life_min"],acquisition_duration_min)
    true_rate = 100_000 * activity * beta_plus_branch * decay_factor * math.exp(-0.12 * activity)
    # Fixed generic scanner peak-rate reference: beta-plus branching then lowers
    # the actual true-pair yield at the same activity, as it should.
    peak_rate = 100_000 * (1 / .12) * math.exp(-1)
    true_counts = max(100, 250_000 * true_rate / peak_rate * acquisition_duration_min/10.0)
    scatter_counts = true_counts * scatter_pct / 100
    random_counts = true_counts * random_pct / 100 * activity / 3
    true_map = truth / truth.sum() * true_counts
    background_pixels = max(int(body.sum()), 1)
    scatter_map = np.zeros_like(truth); random_map = np.zeros_like(truth)
    scatter_map[body] = scatter_counts / background_pixels
    random_map[body] = random_counts / background_pixels
    expected_prompts = true_map + scatter_map + random_map
    rng = np.random.default_rng(seed)
    prompts = rng.poisson(np.maximum(expected_prompts, 0)).astype(float)
    correction = np.zeros_like(truth)
    if apply_scatter: correction += scatter_map
    if apply_random: correction += random_map
    estimate = np.maximum(prompts - correction, 0)
    # Normalise to the known uniform-background concentration, not the whole-body
    # mean (which includes the deliberately hot structures).
    reference_scale = max(float(true_map[bg].mean()), 1e-9)
    estimate = estimate / reference_scale
    reference = true_map / reference_scale
    if psf_model:
        # Approximate PSF-aware recovery for this image-quality illustration.
        # The main reconstruction module uses the PSF inside A@H instead.
        estimate = richardson_lucy_gaussian(estimate, detector_sigma, iterations=4)
    mean_bg = float(estimate[bg].mean())
    cv_bg = float(100 * estimate[bg].std(ddof=1) / max(abs(mean_bg), 1e-9))
    bias = float(100 * (mean_bg - 1.0))
    return {"image": estimate, "reference": reference, "truth": truth, "true_rate": true_rate,
            "scatter_rate": true_rate * scatter_pct/100, "random_rate": true_rate * random_pct/100*activity/3,
            "necr": true_rate**2 / max(true_rate + true_rate*scatter_pct/100 + 2*true_rate*random_pct/100*activity/3, 1),
            "prompts": int(prompts.sum()), "cv": cv_bg, "bias": bias,
            "scatter_counts": scatter_counts, "random_counts": random_counts,
            "isotope": isotope, "beta_plus_branch": beta_plus_branch, "decay_factor": decay_factor,
            "acquisition_duration_min": acquisition_duration_min, "range_sigma_px": range_sigma,
            "effective_resolution_fwhm_mm": math.sqrt(ISOTOPES[isotope]["range_fwhm_mm"]**2 + detector_fwhm_mm**2),
            "physical_resolution_fwhm_mm": math.sqrt(ISOTOPES[isotope]["range_fwhm_mm"]**2 + detector_fwhm_mm**2),
            "ideal_psf_corrected_lower_bound_mm": ISOTOPES[isotope]["range_fwhm_mm"],
            "psf_model": psf_model}


def add_teaching_notes(title: str, teaching: str, equations: list[str], take_home: str):
    with st.expander(f"Teaching notes & equations — {title}", expanded=False):
        st.markdown(teaching)
        for equation in equations:
            st.latex(equation)
        st.info(take_home)


# -----------------------------------------------------------------------------
# Shared controls
# -----------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1> PET Virtual Laboratory</h1>"
    "<p>From positron emission to quantitative imaging · Interactive teaching application</p></div>",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Lecture controls")
    isotope = st.selectbox("Radionuclide", list(ISOTOPES), index=0)
    p = ISOTOPES[isotope]
    st.metric("Half-life", f"{p['half_life_min']:g} min")
    st.metric("Positron endpoint energy", f"{p['endpoint_mev']:.3f} MeV")
    st.metric("Mean positron range in water", f"{p['mean_range_mm']:.1f} mm")
    st.metric("Projected range-kernel FWHM", f"{p['range_fwhm_mm']:.2f} mm")
    st.metric("β⁺ emission probability", f"{100*p['beta_plus_branch']:.2f}%")
    st.caption("Changing isotope selects its own positron-range kernel and β⁺ yield. The scanner detector PSF remains a separate scanner property; both responses combine in the effective imaging response. Annihilation photons remain approximately 511 keV each.")
    st.divider()
    st.caption("Educational simulation only. Not for clinical use or scanner acceptance decisions.")

# Top-level module navigation
modules = st.tabs([
    "Overview", "1 · Positron physics", "2 · PET detection", "3 · Sinograms",
    "4 · Reconstruction", "5 · Quality control", "6 · Knowledge check"
])

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
with modules[0]:
    st.subheader("The PET imaging chain")
    st.write("Follow one radioactive decay event through the scanner and into a quantitative image.")
    cols = st.columns(5)
    items = [
        ("01", "Positron emission", "β⁺ decay and positron energy", "#7656ad"),
        ("02", "Annihilation", "Two approximately 511 keV photons", "#3b83b6"),
        ("03", "Coincidence", "Detector pair and line of response", "#248f79"),
        ("04", "Reconstruction", "Sinogram to image", "#bd842d"),
        ("05", "Quality control", "Accuracy, resolution and noise", "#b65050"),
    ]
    for col, (num, title, desc, colour) in zip(cols, items):
        with col:
            st.markdown(
                f"<div style='background:{colour};color:white;border-radius:12px;padding:13px 10px;min-height:132px'>"
                f"<div style='font-size:12px;opacity:.8'>{num}</div><b>{title}</b>"
                f"<div style='font-size:12px;margin-top:8px'>{desc}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("### Learning outcomes")
    st.markdown(
        "By the end of this session, students should be able to explain positron emission and annihilation; "
        "distinguish true, scattered and random coincidences; interpret a sinogram; describe the principles "
        "of FBP and iterative reconstruction; and identify key PET image-quality and quantitative QC metrics."
    )
    st.info("Teaching prompt: What needs to happen between a positron being emitted and a voxel having a reliable activity concentration?")
    with st.expander("How to use this laboratory after the 30-minute lecture"):
        st.markdown("The lecture sequence is designed for a short guided introduction, but each module contains extended reading, equations and controls for independent study. A useful self-study sequence is: (1) compare F-18 and Ga-68 positron endpoints and positron-range implications; (2) explore direct annihilation, p-Ps and o-Ps pathways; (3) vary scatter, randoms and timing resolution and observe count-rate/image effects; (4) move a point source and trace its sinogram path; (5) compare FBP, MLEM and OSEM at different total subset updates; and (6) select a phantom, inspect the prescribed ROI geometry, calculate metrics and explain which commissioning baseline or accreditation criterion is relevant.")
        st.markdown("Students should save screenshots or export tables for their lab notes, but must not interpret synthetic measurements as clinical scanner performance or formal QC acceptance results.")
    with st.expander("PET acronym glossary"):
        st.dataframe(pd.DataFrame([
            ["β⁺", "Beta-plus / positron emission"], ["LOR", "Line of response"],
            ["TOF", "Time-of-flight"], ["PSF", "Point-spread function"],
            ["PVE", "Partial-volume effect"], ["FBP", "Filtered backprojection"],
            ["MLEM", "Maximum-likelihood expectation maximisation"],
            ["OSEM", "Ordered-subsets expectation maximisation"],
            ["NECR", "Noise-equivalent count rate"], ["CRC", "Contrast recovery coefficient"],
            ["CV", "Coefficient of variation"], ["SOR", "Spillover ratio"],
            ["SUV", "Standardised uptake value"], ["VOI / ROI", "Volume / region of interest"],
            ["RL", "Richardson–Lucy deconvolution"]
        ], columns=["Acronym", "Meaning"]), hide_index=True, use_container_width=True)
        st.caption("‘POV’ is not a standard PET reconstruction term; the likely intended terms in this context are TOF or PVE.")
    add_teaching_notes("The complete PET imaging chain",
        "PET quantification depends on every stage: radionuclide decay and positron transport; photon detection and coincidence classification; geometric encoding in projection data; corrections and image reconstruction; and calibration/QC. Errors introduced upstream propagate into the final image.",
        [r"\text{Decay}\rightarrow\text{transport}\rightarrow\text{detection}\rightarrow\text{reconstruction}\rightarrow\text{quantification}"],
        "A visually attractive PET image is not necessarily quantitatively accurate. Image quality and quantitative validity must be evaluated separately.")

# -----------------------------------------------------------------------------
# Positron physics
# -----------------------------------------------------------------------------
with modules[1]:
    st.subheader("From β⁺ decay to annihilation")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Animated positron history**")
        st.caption("Follow the positron from nuclear emission through stochastic slowing, thermalisation and annihilation.")
        branch_choice = st.selectbox("Show annihilation pathway", [
            "Direct annihilation", "Para-positronium (p-Ps)",
            "Ortho-Ps in tissue (pick-off)", "Ortho-Ps in vacuum (3γ)"
        ], key="positron_pathway")
        ps_fraction = st.slider("Illustrative positronium formation fraction (%)", 0, 60, 40, 1, key="ps_fraction")
        st.components.v1.html(positronium_animation(isotope, branch_choice, ps_fraction), height=390, scrolling=False)
    with right:
        st.markdown("**The essential physics**")
        st.latex(r"p\rightarrow n+e^++\nu_e")
        st.write("In β⁺ decay, a proton is converted into a neutron, emitting a positron and an electron neutrino. The positron slows through interactions with the surrounding medium before annihilating with an electron.")
        st.latex(r"e^++e^-\rightarrow 2\gamma\quad (511\;\mathrm{keV\;each})")
        st.metric("Approximate mean positron range", f"{p['mean_range_mm']:.1f} mm")
        st.caption(f"The animation track length responds to the selected {isotope} range value ({p['mean_range_mm']:.1f} mm). This is visual scaling only; real positron trajectories are stochastic and must be modelled from the full beta spectrum and material transport physics.")
        direct_fraction = 100 - ps_fraction
        pps_fraction = ps_fraction * 0.25
        ops_fraction = ps_fraction * 0.75
        f1, f2, f3 = st.columns(3)
        f1.metric("Direct annihilation", f"{direct_fraction:.1f}%")
        f2.metric("p-Ps", f"{pps_fraction:.1f}%")
        f3.metric("o-Ps", f"{ops_fraction:.1f}%")
        st.caption("Illustrative population bookkeeping only: the overall Ps fraction is a teaching control; the 1:3 p-Ps:o-Ps split is a spin-statistical approximation, not a patient-specific measurement.")
    st.markdown("### Compare common PET radionuclides")
    st.caption("β⁺ emission probabilities are isotope-specific; non-β⁺ branches (for example electron capture) do not produce the positron-annihilation pair used in conventional PET. Decay probabilities sourced from IAEA standard positron-emitter decay data.")
    iso_df = pd.DataFrame([
        {"Radionuclide": k, "Half-life (min)": v["half_life_min"], "β⁺ probability (%)": 100*v["beta_plus_branch"], "Endpoint energy (MeV)": v["endpoint_mev"], "Mean range in water (mm)": v["mean_range_mm"], "Range-kernel FWHM (mm)": v["range_fwhm_mm"]}
        for k, v in ISOTOPES.items()
    ])
    st.dataframe(iso_df, hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3.1))
    ax.bar(iso_df["Radionuclide"], iso_df["Endpoint energy (MeV)"])
    ax.set_ylabel("Positron endpoint energy (MeV)"); ax.set_title("Endpoint energy differs substantially between radionuclides")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with st.expander("Why does positron range matter?"):
        st.write("The annihilation position is displaced from the decay position. This contributes to intrinsic spatial blurring before detector resolution, non-collinearity and reconstruction effects are considered. Higher endpoint energy generally leads to a broader positron range distribution, but endpoint energy alone does not describe the full spectrum or mean range.")
        st.latex(r"\text{Observed resolution}\;\neq\;\text{detector resolution alone}")
    st.warning("Teaching distinction: positron range is not the same as the range of the 511 keV annihilation photons. The photons may travel through the patient and undergo attenuation or scatter.")
    st.markdown("### Extended learning: the complete positron story")
    with st.expander("1. β⁺ nuclear decay and the positron energy spectrum", expanded=True):
        st.markdown("A proton-rich radionuclide can undergo β⁺ decay when the decay is energetically allowed. At the nuclear level, a proton changes into a neutron while a positron and an electron neutrino are emitted. The available decay energy is shared continuously between the positron and neutrino (with a small nuclear-recoil contribution). Therefore, emitted positrons are not monoenergetic: the spectrum extends from near zero to an endpoint energy. The endpoint is not the mean energy.")
        st.latex(r"p\rightarrow n+e^++\nu_e")
        st.latex(r"Q_{\beta^+}=[M(A,Z)-M(A,Z-1)-2m_e]c^2\quad\text{(atomic-mass convention)}")
        st.markdown("The subtraction of two electron masses in the atomic-mass expression accounts for the emitted positron and the change in the atomic electron inventory. β⁺ emission requires an energy difference exceeding 1.022 MeV. Competing electron capture may also occur in some nuclides, so the β⁺ branching fraction matters when relating activity to detected annihilation events.")
    with st.expander("2. Positron transport, energy loss and thermalisation"):
        st.markdown("Immediately after emission, the positron has kinetic energy drawn from the continuous β⁺ spectrum. It travels through tissue or phantom material and undergoes many stochastic interactions. Inelastic collisions transfer energy to electronic excitation and ionisation; elastic Coulomb scattering changes its direction. As its kinetic energy decreases, its path becomes a tortuous random walk rather than a straight line. It eventually reaches near-thermal energies relative to the local medium. Thermalisation does not mean that the positron stops permanently: it continues to diffuse and may annihilate directly or form positronium.")
        st.markdown("**Why this matters for PET:** the annihilation position is displaced from the radionuclide decay position. The spatial distribution of this displacement is the positron-range kernel. It depends on the full emission spectrum and material composition/density—not just the endpoint energy. A higher endpoint energy often produces a broader range distribution, but range values are material- and radionuclide-dependent.")
        st.latex(r"\text{Decay position}\;\xrightarrow{\text{stochastic transport}}\;\text{annihilation position}")
        st.markdown("Positron range is one component of intrinsic spatial blurring. Detector size, acollinearity of annihilation photons, photon non-normal incidence, reconstruction and post-filtering also contribute to the measured system resolution.")
    with st.expander("3. Positronium formation: what is it? Why is it relevant?"):
        st.markdown("A thermalised positron can bind with an electron to form positronium (Ps), an electron–positron bound state. In biological tissue, reports commonly estimate that a substantial fraction—often around 30–40%—of positrons form Ps, but the fraction is not a universal fixed constant. It can depend on material, molecular environment, positron energy, tracer and experimental method. In an unpolarised formation model, spin statistics give the familiar 1:3 para-to-ortho population ratio: approximately 25% para-positronium and 75% ortho-positronium among formed Ps.")
        st.latex(r"e^+ + e^- \rightarrow \mathrm{Ps}")
        st.markdown("""**Para-positronium (p-Ps):** electron and positron spins are antiparallel (singlet, total spin S=0). Its dominant self-annihilation is into two photons. The vacuum mean lifetime is approximately 125 ps.

**Ortho-positronium (o-Ps):** spins are parallel (triplet, S=1). In vacuum, the dominant self-annihilation is into three photons, with a mean lifetime of approximately 142 ns. The three photons share the available energy, so they are not each 511 keV. Four-photon p-Ps and five-photon o-Ps decays are possible higher-order modes but are extremely rare compared with the dominant channels.""")
        st.latex(r"\mathrm{p\!-\!Ps}\;(S=0)\rightarrow 2\gamma")
        st.latex(r"\mathrm{o\!-\!Ps}\;(S=1)\rightarrow 3\gamma\quad\text{(dominant vacuum mode)}")
    with st.expander("4. How positronium decays in tissue: pick-off and spin conversion"):
        st.markdown("The long vacuum lifetime of o-Ps is strongly modified in condensed matter. **Pick-off annihilation** occurs when the positron in o-Ps annihilates with an external electron from a surrounding molecule; this commonly gives a two-photon final state and shortens the observed o-Ps lifetime. **Spin conversion / spin exchange** can convert o-Ps into p-Ps through interactions with the environment, after which rapid two-photon annihilation occurs. Consequently, the observed o-Ps lifetime in tissue is far shorter than the vacuum value and is sensitive to local molecular and nanoscopic environment.")
        st.markdown("The usual PET image is still formed predominantly from pairs of approximately 511-keV photons. Standard clinical PET does not routinely reconstruct positronium lifetime or three-photon events as a separate image. Positronium-lifetime imaging and related approaches are emerging research directions that may add information complementary to conventional activity images.")
        st.latex(r"\mathrm{o\!-\!Ps}+e^-_{\mathrm{molecule}}\rightarrow 2\gamma+e^-_{\mathrm{molecule}}{}'\quad\text{(pick-off, schematic)}")
        st.latex(r"\mathrm{o\!-\!Ps}\xrightarrow{\text{spin conversion}}\mathrm{p\!-\!Ps}\rightarrow 2\gamma")
    with st.expander("5. Direct annihilation, energy and momentum conservation"):
        st.markdown("A positron that does not form Ps can annihilate with an electron. When the pair is approximately at rest in its centre-of-mass frame, two photons are emitted in nearly opposite directions, each with energy close to the electron rest energy. Their directions are not exactly 180° apart because the annihilating pair has residual momentum (acollinearity), which limits PET spatial resolution even for point-like detectors.")
        st.latex(r"e^++e^-\rightarrow 2\gamma,\qquad E_\gamma\simeq m_ec^2=511\;\mathrm{keV}")
        st.latex(r"\Delta\theta\neq 0\quad\text{(small acollinearity)}")
    with st.expander("Further reading: positronium in PET and tissue"):
        st.markdown("- [IAEA standard positron-emitter decay data](https://www-nds.iaea.org/publications/indc/indc-nds-0535.pdf) — positron-emission probabilities and beta endpoint energies.\n- [Positronium lifetime image reconstruction for TOF PET (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9829407/) — overview of Ps formation, p-Ps/o-Ps and tissue interactions.\n- [High-resolution positronium lifetime tomography (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785412/) — discussion of Ps lifetime sensitivity to tissue microenvironment.\n- [PET: its 65 years and beyond (2024)](https://link.springer.com/article/10.1007/s40766-024-00050-3) — PET physics including non-collinearity and Ps.\n- [Direct imaging of three-photon decay in PET (2026)](https://www.nature.com/articles/s42005-026-02782-6) — emerging three-photon positronium imaging concepts.")
    add_teaching_notes("Positron physics",
        "**Core concepts:** β⁺ decay requires sufficient energy to create the positron; the positron has a continuous kinetic-energy spectrum, loses energy by Coulomb interactions and annihilates after slowing. Annihilation at rest produces two 511 keV photons that are approximately, but not perfectly, back-to-back.",
        [r"Q_{\beta^+} = [M(A,Z)-M(A,Z-1)-2m_e]c^2", r"e^+ + e^- \rightarrow 2\gamma", r"E_\gamma \approx 511\;\mathrm{keV}"],
        "Higher positron energies generally broaden the positron-range contribution. The endpoint energy is not the mean energy and does not alone determine the full range distribution.")

# -----------------------------------------------------------------------------
# Detector and coincidence physics
# -----------------------------------------------------------------------------
with modules[2]:
    st.subheader("PET detection and coincidence classification")
    st.write("An annihilation pair may be recorded as a prompt coincidence, but not every prompt is a true event.")
    event_type = st.radio("Select an event type", ["True", "Scattered", "Random", "Single"], horizontal=True)
    fig = detector_3d_figure(event_type)
    diagram_col, explanation_col = st.columns([0.85, 1.15], gap="medium")
    with diagram_col:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    descriptions = {
        "True": "Both annihilation photons are detected within the timing and energy windows without significant direction-changing interactions. The connecting line is the line of response (LOR).",
        "Scattered": "At least one photon changes direction before detection. The detected pair defines an incorrect apparent LOR and contributes to image degradation unless corrected or modelled.",
        "Random": "Two unrelated photons from different annihilations are detected within the coincidence timing window and paired by the system.",
        "Single": "Only one member of an annihilation pair is detected. Singles are not, by themselves, a coincidence event.",
    }
    with explanation_col:
        st.info(descriptions[event_type])
        event_facts = {
            "True": "Valid prompt candidate: both photons originate from the same annihilation and define the correct LOR (within acollinearity and detector-position uncertainty).",
            "Scattered": "Prompt but spatially mispositioned: the dashed apparent LOR does not pass through the true annihilation point. Energy-window rejection and scatter correction/model-based estimation help manage these events.",
            "Random": "Accidental prompt: photons from different annihilations arrive within the coincidence timing window. Randoms can be estimated using a delayed coincidence window or singles-based methods.",
            "Single": "Not a coincidence: only one photon is detected. Singles contribute to random-coincidence rates, but are not themselves prompt pairs."
        }
        st.markdown(event_facts[event_type])
        st.caption("Top-down detector-ring schematic. Colours identify photon origin/path; arrows indicate photon travel. Geometry is illustrative, not to scale.")
    st.markdown("### Count-rate model — explore the effect on image quality")
    c1, c2, c3 = st.columns(3)
    with c1: activity = st.slider("Activity (relative units)", 0.1, 10.0, 3.0, 0.1, key="cr_activity")
    with c2: scatter_pct = st.slider("Scatter fraction (%)", 0, 60, 25, 1, key="cr_scatter")
    with c3: random_strength = st.slider("Random contribution (%)", 0, 80, 15, 1, key="cr_random")
    cr_duration = st.slider("Acquisition duration (min)",1,60,10,1,key="cr_duration")
    corr1, corr2, corr3 = st.columns(3)
    with corr1: apply_scatter = st.checkbox("Apply scatter correction", value=True, key="cr_sc_corr")
    with corr2: apply_random = st.checkbox("Apply randoms correction", value=True, key="cr_rnd_corr")
    with corr3: cr_psf = st.checkbox("Use PSF-aware restoration (RL demo)", value=False, key="cr_psf_model")
    cr_detector_fwhm = st.slider("Detector/system FWHM (mm)", 1.0, 8.0, 4.0, 0.25, key="cr_detector_fwhm")
    cr = simulate_count_rate_image(activity, scatter_pct, random_strength, apply_scatter, apply_random,
                                   isotope=isotope, detector_fwhm_mm=cr_detector_fwhm, psf_model=cr_psf,
                                   acquisition_duration_min=cr_duration)
    st.session_state["count_rate_result"] = cr
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True rate", f"{cr['true_rate']:,.0f} cps")
    m2.metric("Scatter rate", f"{cr['scatter_rate']:,.0f} cps")
    m3.metric("Random rate", f"{cr['random_rate']:,.0f} cps")
    m4.metric("NECR", f"{cr['necr']:,.0f} cps")
    # Show complete count-rate curves and the selected operating point.
    aa = np.linspace(.1, 10, 100)
    curve_decay_factor = decay_average_activity_factor(p["half_life_min"], cr_duration)
    tt = 100_000 * p["beta_plus_branch"] * curve_decay_factor * aa * np.exp(-.12 * aa)
    ss = tt * scatter_pct / 100
    rrates = tt * random_strength / 100 * aa / 3
    nn = tt**2 / np.maximum(tt + ss + 2*rrates, 1)
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.plot(aa, tt, label="True", lw=2); ax.plot(aa, ss, label="Scatter", lw=1.7)
    ax.plot(aa, rrates, label="Random", lw=1.7); ax.plot(aa, nn, label="NECR", lw=2, ls="--")
    ax.axvline(activity, color="black", alpha=.5, ls=":", label="Selected activity")
    ax.set_xlabel("Relative activity"); ax.set_ylabel("Illustrative rate (cps)"); ax.grid(alpha=.2)
    ax.legend(ncol=5, fontsize=8, loc="upper center"); fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("**What happens to the image?** The phantom below is regenerated from the selected count-rate, scatter and random settings.")
    imgcols = st.columns([1, 1.15])
    with imgcols[0]:
        st.pyplot(fig_image(cr["image"], "Simulated PET image after selected corrections", cmap="magma"), use_container_width=True); plt.close("all")
    with imgcols[1]:
        q1, q2, q3 = st.columns(3)
        q1.metric("Background CV", f"{cr['cv']:.1f}%")
        q2.metric("Background bias", f"{cr['bias']:+.1f}%")
        q3.metric("Prompt counts", f"{cr['prompts']:,}")
        st.caption(f"{isotope}: β⁺ probability {100*p['beta_plus_branch']:.2f}%; mean activity during this {cr_duration}-min scan is {100*cr['decay_factor']:.1f}% of the starting activity. Physical effective spatial FWHM estimate ≈ {cr['physical_resolution_fwhm_mm']:.1f} mm. The event yield scales with β⁺ probability and decay during acquisition; the image includes isotope-dependent range blur plus detector blur. The PSF option applies a short Gaussian-kernel Richardson–Lucy restoration to the estimated image; it is an educational deblurring demonstration, not the full iterative system-matrix approach used in the reconstruction module. QC independently regenerates its selected phantom using the same radionuclide and resolution settings.")
    with st.expander("How to interpret the count-rate curves and isotope dependence"):
        st.markdown("- At a fixed activity (Bq), integrated positron yield depends on β⁺ emission probability, mean activity during the scan and acquisition duration. The simulation scales its true-event rate and Poisson event yield by that branch; electron-capture decays do not generate the positron-annihilation pair used by conventional PET.\n- Prompt-gamma cascades and their additional scatter/random coincidence contributions are not modelled here.\n- At low activity, increasing activity increases true events and improves counting statistics.\n- At higher activity, the illustrative dead-time/count-loss term makes the true rate roll over.\n- Scatter adds mispositioned events; randoms add unrelated coincidences. Both increase prompts without adding useful true information.\n- The isotope changes the positron-range kernel in the simulated image using its own projected range FWHM. Higher-range emitters blur small structures more and generally reduce their recovery. It does not change the 511-keV annihilation photon energy.\n- PSF resolution modelling includes a detector-response model in iterative reconstruction; the count-rate and QC exercises use a short Richardson–Lucy restoration demonstration with a known Gaussian detector PSF. Neither can remove the irreducible positron-range and acollinearity components. Improved apparent recovery can also increase noise or edge overshoot (Gibbs artefacts).\n- NECR is a comparative count-statistics metric, not a complete predictor of clinical image quality.")
        st.latex(r"FWHM_{eff}^2\approx FWHM_{range}^2+FWHM_{detector}^2")
        st.latex(r"\bar A/A_0=(1-e^{-\lambda T})/(\lambda T),\quad \lambda=\ln2/T_{1/2}")
        st.latex(r"P=T+S+R")
        st.latex(r"NECR=\frac{T^2}{T+S+2R}")
        st.latex(r"CV(\%)=100\frac{\sigma_{background}}{\mu_{background}}")
    st.markdown("### Time-of-flight localisation")
    timing_ps = st.slider("Coincidence timing resolution (FWHM)", 50, 1000, 400, 10, key="tof_timing")
    localisation_cm = 0.0149896 * timing_ps
    st.metric("TOF localisation uncertainty (FWHM)", f"{localisation_cm:.2f} cm")
    st.latex(r"\Delta x=\frac{c\,\Delta t}{2}")
    st.write("Better timing narrows the probability distribution along the LOR. It does not directly change detector crystal size or eliminate positron-range blurring.")
    with st.expander("Extended learning: detector physics, event processing and corrections"):
        st.markdown("**Scintillation detection:** a 511-keV photon interacts in a scintillator, most commonly by photoelectric absorption or Compton scattering. Deposited energy excites the scintillator and produces light. A photomultiplier tube or silicon photomultiplier (SiPM) converts the light into an electrical pulse. The detector electronics estimate event energy and arrival time; energy and timing windows are then used to identify candidate coincident events.")
        st.markdown("**Line of response (LOR):** a pair of detectors registering events within the coincidence window defines a geometric LOR. Conventional non-TOF PET assigns the event to the LOR; TOF PET additionally estimates a probability distribution for the annihilation location along it. A coincidence event is not automatically true: scattered and random events also enter the prompt data.")
        st.markdown("**Corrections:** normalisation compensates for detector-pair sensitivity differences; dead-time correction addresses count losses when electronics or detectors cannot process events; random correction estimates unrelated pairs; scatter correction estimates photons whose directions changed; attenuation correction accounts for photons removed from the detected line. These corrections influence both image appearance and quantitative accuracy.\n\n**Resolution effects:** positron range depends on radionuclide and material; photon acollinearity depends on residual momentum of the annihilating pair; detector response depends on crystal size, penetration and depth of interaction. These combine with reconstruction and voxel sampling to determine the measured spatial resolution. PSF modelling places a model of detector/system response inside the iterative reconstruction. TOF adds timing-based localisation along the LOR. PVE is the under-recovery of activity in structures comparable to or smaller than system resolution.")
        st.latex(r"FWHM_{eff}^2\approx FWHM_{range}^2+FWHM_{detector}^2+FWHM_{acollinearity}^2")
        st.markdown("**Count-rate behaviour:** as activity rises, true, scatter and random rates rise, while dead-time losses become increasingly important. The activity giving the highest NECR is not necessarily the activity giving the best result for every imaging task. NECR is a standardised comparative measure of statistical performance, not a complete model of image quality.")
        st.latex(r"P=T+S+R")
        st.latex(r"NECR=\frac{T^2}{T+S+2kR}\quad (k=1\text{ for the common randoms convention})")
        st.latex(r"R_{\mathrm{random}}\propto S_1S_2\,\Delta t")
        st.latex(r"\Delta x_{\mathrm{TOF}}=\frac{c\,\Delta t}{2};\quad \sigma_x\simeq\frac{c\,\Delta t_{\mathrm{FWHM}}}{2.355\times2}")
        st.caption("The displayed localisation relation uses FWHM timing resolution; the spatial FWHM along the LOR is cΔt/2. It is not the total reconstructed image resolution.")
    add_teaching_notes("PET detection and count rates",
        "**Core concepts:** scintillation converts deposited photon energy into light; photosensors convert light into electrical signals. Energy and timing windows select candidate events. A true pair estimates an LOR; scatter changes photon direction; randoms pair unrelated detections.",
        [r"P=T+S+R", r"NECR=\frac{T^2}{T+S+2R}", r"R_{random} \propto S_1S_2\Delta t", r"\Delta x=\frac{c\Delta t}{2}"],
        "Wider timing windows generally increase random coincidences. NECR is a useful count-rate indicator, but image quality also depends on attenuation, resolution, corrections, reconstruction and task.")

# -----------------------------------------------------------------------------
# Sinograms
# -----------------------------------------------------------------------------
with modules[3]:
    st.subheader("From source position to sinogram")
    st.write("A sinogram stores projection measurements as a function of detector position and projection angle. The shared isotope selection changes the simulated positron-range blur before projection.")
    phantom_kind = st.selectbox("Choose a digital phantom", ["NEMA-style hot-sphere phantom", "Uniform cylinder", "Two hot sources", "Shepp-Logan-like anatomy"], index=2, key="sino_phantom")
    two_sep_sino, two_diam_sino, two_diam_right_sino, two_hot_sino, two_hot_right_sino = 8.0, 3.0, 5.0, 4.0, 8.0
    if phantom_kind == "Two hot sources":
        st.markdown("**Asymmetric two-source experiment: different sizes and activities**")
        sctl1, sctl2, sctl3, sctl4, sctl5 = st.columns(5)
        with sctl1: two_diam_sino = st.slider("Left diameter (mm)", 1.5, 6.0, 3.0, 0.5, key="sino_left_diam_v14")
        with sctl2: two_diam_right_sino = st.slider("Right diameter (mm)", 1.5, 6.0, 5.0, 0.5, key="sino_right_diam_v14")
        sino_min_sep = float((two_diam_sino + two_diam_right_sino)/2.0 + 0.5)
        if st.session_state.get("sino_two_sep_v14", max(8.0, sino_min_sep)) < sino_min_sep:
            st.session_state["sino_two_sep_v14"] = sino_min_sep
        with sctl3: two_sep_sino = st.slider("Centre separation (mm)", sino_min_sep, 14.0, max(8.0, sino_min_sep), 0.5, key="sino_two_sep_v14")
        with sctl4: two_hot_sino = st.slider("Left activity / background", 2.0, 12.0, 4.0, 0.5, key="sino_two_hot_v14")
        with sctl5: two_hot_right_sino = st.slider("Right activity / background", 2.0, 12.0, 8.0, 0.5, key="sino_two_hot_right_v14")
        st.caption("Default: 3.0-mm left source at 4× background and 5.0-mm right source at 8× background, with 8-mm centre spacing. The independent diameters and activities make size-dependent recovery and unequal peak amplitudes visible. Use the separation control to explore partial merging; PSF modelling cannot guarantee recovery below the physical resolution limit.")
    n_angles = st.slider("Number of projection angles", 30, 120, 72, 6, key="sino_angles")
    counts = st.slider("10-min reference count scale (before isotope/decay factors)", 20_000, 500_000, 300_000, 10_000, key="sino_counts")
    sino_detector_fwhm = st.slider("Scanner detector/system PSF FWHM (mm)",0.5,8.0,2.0,0.25,key="sino_det_fwhm")
    sino_duration = st.slider("Acquisition duration (min)",1,60,10,1,key="sino_duration")
    sino_c1, sino_c2 = st.columns(2)
    with sino_c1: sino_scatter_fraction = st.slider("Scatter fraction in projections",0.0,0.5,0.05,0.01,key="sino_scatter")
    with sino_c2: sino_random_fraction = st.slider("Random fraction in projections",0.0,0.5,0.03,0.01,key="sino_random")
    sino_decay_factor=decay_average_activity_factor(p["half_life_min"],sino_duration)
    phantom = (make_two_source_phantom(64,two_sep_sino,two_diam_sino,two_hot_sino,
                                       right_hot_activity=two_hot_right_sino,
                                       right_diameter_mm=two_diam_right_sino)
               if phantom_kind == "Two hot sources" else make_phantom(phantom_kind, 64))
    sino_range_sigma = isotope_range_sigma_px(isotope, pixel_mm=1.0)
    sino_prompts, sino_true, sino_scatter, sino_randoms, sino_A = simulate_sinogram(
        phantom, n_angles, 64, counts, sino_scatter_fraction, sino_random_fraction, seed=31,
        range_sigma_px=sino_range_sigma, detector_psf_sigma_px=sino_detector_fwhm/2.355,
        positron_branch=p["beta_plus_branch"],decay_factor=sino_decay_factor,
        acquisition_duration_min=sino_duration
    )
    sino = sino_prompts.reshape(n_angles, 64).T
    a, b = st.columns([1, 1.45])
    with a:
        st.pyplot(fig_image(phantom, "Digital source phantom")); plt.close("all")
    with b:
        st.pyplot(fig_sinogram(sino, "Measured sinogram · positron range + detector PSF")); plt.close("all")
    if phantom_kind == "Two hot sources":
        st.markdown("### What happens to two nearby sources in sinogram space?")
        # Use an identical seed and source distribution; only detector PSF differs.
        ideal_prompts, *_ = simulate_sinogram(
            phantom,n_angles,64,counts,sino_scatter_fraction,sino_random_fraction,seed=31,range_sigma_px=sino_range_sigma,
            detector_psf_sigma_px=0.0,positron_branch=p["beta_plus_branch"],
            decay_factor=sino_decay_factor,acquisition_duration_min=sino_duration)
        ideal_sino=ideal_prompts.reshape(n_angles,64).T
        sfig, sax=plt.subplots(1,2,figsize=(10,3.1))
        vmax=max(float(np.percentile(sino,99.5)),1.0)
        sax[0].imshow(ideal_sino,aspect="auto",origin="lower",cmap="viridis",vmin=0,vmax=vmax)
        sax[0].set_title("Idealised detector · PSF = 0")
        sax[1].imshow(sino,aspect="auto",origin="lower",cmap="viridis",vmin=0,vmax=vmax)
        sax[1].set_title(f"Detector response included · FWHM {sino_detector_fwhm:.1f} mm")
        for ax_s in sax: ax_s.set_xlabel("Projection angle index");ax_s.set_ylabel("Detector bin")
        sfig.tight_layout();st.pyplot(sfig,use_container_width=True);plt.close(sfig)
        st.pyplot(fig_two_source_sinogram_paths(two_sep_sino,n_angles,64,sino_detector_fwhm,
                                                 two_hot_sino,two_hot_right_sino,
                                                 two_diam_sino,two_diam_right_sino),use_container_width=True);plt.close("all")
        st.markdown("The two source centres generate sinusoidal tracks. Their locations are symmetric, but their activity amplitudes are deliberately unequal (left versus right). As the sources move closer, the tracks approach and overlap near the projection angle where their radial coordinates coincide. Detector response and positron range broaden the tracks; this is the projection-space counterpart of image blurring. The left panel is an idealised detector with no detector PSF; the right panel includes the detector response. Both use the same scatter/random fractions and count scale.")
    st.markdown("### Independent point-source geometry demonstrator")
    px, py = st.columns(2)
    with px: source_x = st.slider("Source x position", -0.6, 0.6, 0.25, 0.05)
    with py: source_y = st.slider("Source y position", -0.6, 0.6, -0.15, 0.05)
    angle_deg = st.slider("Projection angle θ", 0, 180, 45, 1)
    point_fig, s_coord_mm = fig_point_projection(source_x, source_y, angle_deg)
    st.metric("Detector offset at selected angle", f"{s_coord_mm:+.2f} mm")
    st.pyplot(point_fig, use_container_width=True); plt.close(point_fig)
    st.latex(r"s=x\cos\theta+y\sin\theta")
    st.write("These controls now move a separate teaching point source and update both the object-space projection line and its sinusoidal trajectory in the sinogram. They do not move the two-hot-source phantom above; that phantom is controlled by its own separation and activity sliders.")
    st.info("Symmetry explains why the two equal-position tracks can meet at particular angles, but it was not the reason the old controls appeared inactive: previously the sliders only updated a number, without updating a plotted point or projection. The new plots make the geometric effect visible.")
    st.caption(f"The selected {isotope} contributes a β⁺ branching probability of {100*p['beta_plus_branch']:.2f}%, mean scan activity factor {100*sino_decay_factor:.1f}% and illustrative positron-range blur; detector/system FWHM is {sino_detector_fwhm:.2f} mm. Both spatial effects are applied before projection. Real PET systems use ring geometry, detector-pair bins, corrections and scanner-specific geometry.")
    with st.expander("Extended learning: projection data, sampling and the inverse problem"):
        st.markdown("A PET acquisition can be stored as list-mode events or histogrammed into projection bins. Histogramming groups events by detector pair, time-of-flight bin (when available), and possibly energy or other coordinates. A sinogram is a convenient 2-D representation of a set of parallel projections: each row or column corresponds to a projection angle, and the other coordinate represents detector offset. A point source traces a sinusoid because its projected coordinate changes with angle.")
        st.markdown("The Radon transform describes the ideal projection of an activity distribution. The inverse problem is to estimate the underlying image from finite, noisy projections. Limited angular sampling, finite detector bins, statistical noise and incomplete or inconsistent measurements all affect the result. Too few views can create streaks and aliasing; too few counts increase noise.")
        st.latex(r"p(s,\theta)=\iint f(x,y)\,\delta(s-x\cos\theta-y\sin\theta)\,dx\,dy")
        st.markdown("**Key interpretation:** the sinogram is measured projection data—not a blurred version of the final image. Reconstruction estimates the image whose projections best explain those data under a chosen system model and noise model.")
        if phantom_kind == "Two hot sources":
            st.markdown(r"**Two-source interpretation:** each point-like source generates a sinusoidal trajectory, with radial coordinate $s(\theta)=x\cos\theta+y\sin\theta$. For two symmetric sources at $x=\pm d/2$, their ideal trajectories are $s_{1,2}(\theta)=\mp(d/2)\cos\theta$. They cross at 90° for this horizontal arrangement. Finite source diameter, positron range and detector PSF broaden the trajectories, making nearby sources less distinguishable in projection data.")
            st.latex(r"s_{1,2}(\theta)=\pm\frac{d}{2}\cos\theta")
            st.markdown("The detector PSF is present in the measured data whether or not it is modelled during reconstruction. In this teaching model it broadens detector-offset bins separately for each projection angle. PSF-aware reconstruction incorporates this same projection-space response in the forward matrix. PSF-aware reconstruction does not alter the original sinogram; it uses a more realistic forward model to estimate the underlying source distribution. It may improve apparent separation and quantitative recovery, but can increase noise or ringing and cannot restore information lost to positron range or inadequate counts.")
    add_teaching_notes("Sinograms and projection data",
        "**Core concepts:** each detector pair defines a projection bin. Across projection angles, a point source traces a sinusoidal path in sinogram space. Projection data are the measurements that reconstruction seeks to explain.",
        [r"s=x\cos\theta+y\sin\theta", r"p(s,\theta)=\iint f(x,y)\,\delta(s-x\cos\theta-y\sin\theta)\,dx\,dy"],
        "The centre of the field of view maps to the central detector coordinate for every angle. Real PET sinograms are organised in scanner-specific ring-difference and detector-pair coordinates.")

# -----------------------------------------------------------------------------
# Reconstruction laboratory
# -----------------------------------------------------------------------------
with modules[4]:
    st.subheader("PET image reconstruction laboratory")
    st.write("Generate projection data and reconstruct the same phantom using different methods.")
    st.markdown("### Where resolution effects enter the PET imaging chain")
    st.pyplot(fig_resolution_path(), use_container_width=True); plt.close("all")
    st.caption("The diagram separates physical blurring from reconstruction modelling. PSF modelling can compensate for the detector/system response when that response is represented adequately in the system matrix; it cannot undo positron transport before annihilation. PVE is the resulting under-recovery when activity spills beyond a small structure's ROI. TOF adds localisation information along the LOR; it is not a substitute for PSF modelling.")
    col1, col2, col3 = st.columns(3)
    with col1:
        recon_phantom_name = st.selectbox("Phantom", ["NEMA-style hot-sphere phantom", "Uniform cylinder", "Two hot sources", "Shepp-Logan-like anatomy"], index=2, key="recon_phantom")
    with col2:
        algorithm = st.selectbox("Reconstruction algorithm", ["Simple backprojection", "Filtered backprojection (FBP)", "MLEM", "OSEM"], index=3)
    with col3:
        iterations = st.slider("Iterations", 1, 40, 8, disabled=(algorithm not in ("MLEM", "OSEM")))
    two_sep_recon, two_diam_recon, two_diam_right_recon, two_hot_recon, two_hot_right_recon = 8.0, 3.0, 5.0, 4.0, 8.0
    if recon_phantom_name == "Two hot sources":
        st.markdown("### Unequal-size, unequal-activity source separation and quantification")
        rctl1,rctl2,rctl3,rctl4,rctl5=st.columns(5)
        with rctl1: two_diam_recon=st.slider("Left source diameter (mm)",1.5,6.0,3.0,0.5,key="recon_left_diam_v14")
        with rctl2: two_diam_right_recon=st.slider("Right source diameter (mm)",1.5,6.0,5.0,0.5,key="recon_right_diam_v14")
        recon_min_sep=float((two_diam_recon+two_diam_right_recon)/2.0+0.5)
        if st.session_state.get("recon_two_sep_v14", max(8.0, recon_min_sep)) < recon_min_sep:
            st.session_state["recon_two_sep_v14"] = recon_min_sep
        with rctl3: two_sep_recon=st.slider("Centre separation (mm)",recon_min_sep,14.0,max(8.0,recon_min_sep),0.5,key="recon_two_sep_v14")
        with rctl4: two_hot_recon=st.slider("Left activity / background",2.0,12.0,4.0,0.5,key="recon_two_hot_v14")
        with rctl5: two_hot_right_recon=st.slider("Right activity / background",2.0,12.0,8.0,0.5,key="recon_two_hot_right_v14")
        st.caption("Default experiment: 3.0-mm left source at 4× background and 5.0-mm right source at 8× background, with 8-mm centre spacing. Different source diameters give different partial-volume recovery; unequal activity exposes spill-in and cross-talk. Compare PSF-aware and PSF-unaware reconstructions using identical prompts.")
    n_subsets = st.select_slider("OSEM subsets", options=[1, 2, 4, 6, 8, 12], value=6, disabled=(algorithm != "OSEM"))
    col4, col5, col6 = st.columns(3)
    with col4: views = st.slider("Projection angles", 30, 120, 72, 6, key="recon_views")
    with col5: total_counts = st.slider("10-min reference count scale (before isotope/decay factors)", 20_000, 500_000, 500_000, 10_000, key="recon_counts")
    with col6: post_sigma = st.slider("Post-filter σ (pixels)", 0.0, 2.0, 0.4, 0.1)
    recon_duration=st.slider("Acquisition duration (min)",1,60,10,1,key="recon_duration")
    recon_decay_factor=decay_average_activity_factor(p["half_life_min"],recon_duration)
    if algorithm == "MLEM":
        st.caption(f"MLEM: {iterations} full-data update(s). Each update uses all {views} projection views. Subsets are not used.")
    elif algorithm == "OSEM":
        st.caption(f"OSEM: {iterations} nominal iteration(s) × {n_subsets} subsets = {iterations*n_subsets} sequential subset updates. One subset is the MLEM special case.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1: scatter_frac = st.slider("Scatter fraction", 0.0, 0.5, 0.03, 0.01)
    with sc2: random_frac = st.slider("Random fraction", 0.0, 0.5, 0.02, 0.01)
    with sc3: detector_fwhm = st.slider("Scanner detector/system PSF FWHM (mm)", 0.5, 8.0, 2.0, 0.25, key="recon_detector_fwhm")
    resolution_c1, resolution_c2 = st.columns(2)
    with resolution_c1:
        psf_model = st.checkbox("Model scanner detector PSF in MLEM/OSEM", value=False,
                                disabled=(algorithm not in ("MLEM","OSEM")), key="recon_psf_model")
    with resolution_c2:
        range_model = st.checkbox("Model selected isotope's positron-range kernel", value=False,
                                  disabled=(algorithm not in ("MLEM","OSEM")), key="recon_range_model")
    st.caption("Important distinction: detector PSF FWHM is a scanner property and is therefore the same for all isotopes on that scanner. The positron-range kernel is isotope-specific. The measurements always contain both physical effects; these two controls independently add their known response operators to the iterative reconstruction model.")
    psf_model = bool(psf_model and algorithm in ("MLEM","OSEM"))
    range_model = bool(range_model and algorithm in ("MLEM","OSEM"))
    pixel_mm = 1.0
    range_sigma = isotope_range_sigma_px(isotope, pixel_mm)
    detector_sigma = detector_fwhm/2.355/pixel_mm
    isotope_range_fwhm = float(p["range_fwhm_mm"])
    eff_fwhm_no = math.sqrt(isotope_range_fwhm**2 + detector_fwhm**2)
    r1,r2,r3=st.columns(3)
    r1.metric("Selected isotope range-kernel FWHM",f"{isotope_range_fwhm:.2f} mm")
    r2.metric("Combined physical FWHM estimate",f"{eff_fwhm_no:.2f} mm")
    r3.metric("Scanner detector PSF FWHM",f"{detector_fwhm:.2f} mm")
    st.pyplot(fig_isotope_kernel_comparison(isotope, detector_fwhm), use_container_width=True); plt.close("all")
    with st.expander("Why F-18 and Ga-68 may look similar on a clinical scanner"):
        f18 = ISOTOPES["F-18"]["range_fwhm_mm"]
        ga68 = ISOTOPES["Ga-68"]["range_fwhm_mm"]
        clinical_det = 4.5
        preclinical_det = 1.5
        clinical_f18 = math.sqrt(f18**2 + clinical_det**2)
        clinical_ga68 = math.sqrt(ga68**2 + clinical_det**2)
        preclinical_f18 = math.sqrt(f18**2 + preclinical_det**2)
        preclinical_ga68 = math.sqrt(ga68**2 + preclinical_det**2)
        st.markdown("The detector response can dominate the combined resolution budget. On a clinical system with a relatively broad detector/system response, the difference between F-18 and Ga-68 is real but can look modest. On a high-resolution preclinical system, the same isotope-specific positron-range difference becomes a much larger fraction of the total blur.")
        f1, f2 = st.columns(2)
        f1.metric("Clinical example · F-18 / Ga-68", f"{clinical_f18:.2f} / {clinical_ga68:.2f} mm")
        f2.metric("Preclinical example · F-18 / Ga-68", f"{preclinical_f18:.2f} / {preclinical_ga68:.2f} mm")
        st.latex(r"\sigma_{range}=\frac{FWHM_{range}}{2.355},\qquad \sigma_{det}=\frac{FWHM_{det}}{2.355}")
        st.latex(r"FWHM_{eff}\approx\sqrt{FWHM_{range}^2+FWHM_{det}^2}")
        st.markdown("The first checkbox models the scanner detector response with the FWHM set by the slider. The second checkbox models the selected radionuclide's range-response kernel. When both are enabled, the forward model uses the combined response (detector response × geometric projector × isotope range response). This is a teaching correction experiment; routine clinical reconstructions do not universally apply positron-range correction.")
        st.caption("Illustrative Gaussian quadrature only: these numbers are not scanner measurements. The detector FWHM must be measured for the scanner and reconstruction protocol being taught.")
    st.caption(f"Selected isotope: {isotope}. Mean positron range in water = {p['mean_range_mm']:.1f} mm; representative projected range-kernel FWHM = {isotope_range_fwhm:.2f} mm. At the current {detector_fwhm:.2f}-mm scanner PSF, F-18 and Ga-68 may look relatively similar on a clinical-resolution system because detector blur dominates. The difference becomes more visible on a high-resolution preclinical system. The range kernel is Gaussian-equivalent, not a full tissue-specific Monte Carlo transport kernel.")
    with st.expander("Interactive PVE (partial-volume effect) demonstration"):
        st.markdown("PVE occurs when the reconstructed resolution is comparable to the structure size. Activity spreads outside the anatomical ROI (spill-out) and neighbouring background spills into it (spill-in), so the measured mean concentration is biased. The following idealised disk experiment calculates mean recovery inside the original ROI after Gaussian blurring.")
        pve_diameter_mm = st.slider("Hot structure diameter (mm)", 1.0, 20.0, 5.0, 0.5, key="pve_diameter")
        pve_fwhm_mm = math.sqrt(p["range_fwhm_mm"]**2 + detector_fwhm**2 + 0.5**2)
        pve_pixel_mm = 0.25
        pve_n = 128
        pve_y,pve_x=np.mgrid[-pve_n//2:pve_n//2,-pve_n//2:pve_n//2]
        pve_radius_px=(pve_diameter_mm/2)/pve_pixel_mm
        pve_true=((pve_x**2+pve_y**2)<=pve_radius_px**2).astype(float)
        pve_sigma_px=(pve_fwhm_mm/2.355)/pve_pixel_mm
        pve_blurred=gaussian_filter(pve_true,pve_sigma_px,mode="constant")
        pve_roi=pve_true>0
        pve_recovery=100*float(np.mean(pve_blurred[pve_roi])) if np.any(pve_roi) else 0.0
        pv1,pv2,pv3=st.columns(3)
        pv1.metric("Approximate system FWHM",f"{pve_fwhm_mm:.2f} mm")
        pv2.metric("Structure diameter",f"{pve_diameter_mm:.1f} mm")
        pv3.metric("Mean activity recovery",f"{pve_recovery:.1f}%")
        pfig,paxs=plt.subplots(1,3,figsize=(10,3))
        paxs[0].imshow(pve_true,cmap="inferno",origin="lower");paxs[0].set_title("True activity")
        paxs[1].imshow(pve_blurred,cmap="inferno",origin="lower",vmin=0,vmax=1);paxs[1].set_title("After system blur")
        paxs[2].plot(pve_true[pve_n//2],label="True profile");paxs[2].plot(pve_blurred[pve_n//2],label="Blurred profile")
        paxs[2].set_title("Central line profile");paxs[2].legend(fontsize=7);paxs[2].grid(alpha=.2)
        for axp in paxs[:2]: axp.set_xticks([]);axp.set_yticks([])
        pfig.tight_layout();st.pyplot(pfig,use_container_width=True);plt.close(pfig)
        curve_diameters=np.arange(1.0,20.5,1.0);curve_recovery=[]
        for diameter in curve_diameters:
            radius=(diameter/2)/pve_pixel_mm
            disk=((pve_x**2+pve_y**2)<=radius**2).astype(float)
            blurred_disk=gaussian_filter(disk,pve_sigma_px,mode="constant")
            curve_recovery.append(100*float(np.mean(blurred_disk[disk>0])))
        pvfig,pvax=plt.subplots(figsize=(7,2.6))
        pvax.plot(curve_diameters,curve_recovery,marker="o",label="Mean ROI recovery")
        pvax.axvline(pve_diameter_mm,color="#b45b43",ls="--",label=f"Selected {pve_diameter_mm:g} mm")
        pvax.axhline(100,color="grey",ls=":",lw=1);pvax.set_xlabel("Hot structure diameter (mm)");pvax.set_ylabel("Mean activity recovery (%)")
        pvax.grid(alpha=.2);pvax.legend();pvfig.tight_layout();st.pyplot(pvfig,use_container_width=True);plt.close(pvfig)
        st.markdown("**Interpretation:** smaller structures relative to system FWHM have lower recovery. This simplified disk model isolates PVE; it is not a clinical recovery-coefficient calibration and does not include noise, motion, nonuniform activity or a measured spatially variant PSF.")
    run_recon = st.button("Run reconstruction", type="primary", use_container_width=True)
    recon_key = (isotope, recon_phantom_name, two_sep_recon, two_diam_recon, two_diam_right_recon,
                 two_hot_recon, two_hot_right_recon, algorithm, iterations, n_subsets, views,
                 total_counts, post_sigma, scatter_frac, random_frac, detector_fwhm, psf_model, range_model, recon_duration)
    if run_recon or "recon_result" not in st.session_state or st.session_state.get("recon_key") != recon_key:
        with st.spinner("Simulating projection data and reconstructing image..."):
            truth_image = (make_two_source_phantom(64,two_sep_recon,two_diam_recon,two_hot_recon,
                                                   right_hot_activity=two_hot_right_recon,
                                                   right_diameter_mm=two_diam_right_recon)
                           if recon_phantom_name == "Two hot sources" else make_phantom(recon_phantom_name, 64))
            prompts, true_c, sc, rnd, A = simulate_sinogram(
                truth_image, views, 64, total_counts, scatter_frac, random_frac, seed=123,
                range_sigma_px=range_sigma, detector_psf_sigma_px=detector_sigma,
                use_psf_model=psf_model, positron_branch=p["beta_plus_branch"],
                decay_factor=recon_decay_factor, acquisition_duration_min=recon_duration,
                model_range_response=range_model
            )
            raw_sino = prompts.reshape(views, 64).T
            H_range = build_image_range_operator(64, round(float(range_sigma), 3))
            range_blurred_image = (H_range @ truth_image.ravel()).reshape(truth_image.shape)
            # If range response is modelled in the forward system, the desired
            # estimate is the original source distribution. Otherwise the target
            # remains the physically range-blurred annihilation distribution.
            target_image = truth_image.copy() if range_model else range_blurred_image.copy()
            rec = reconstruct(prompts, A, truth_image.shape, algorithm, iterations, sc, rnd, n_subsets=n_subsets)
            if post_sigma > 0:
                rec = gaussian_filter(rec, post_sigma)
            st.session_state["recon_result"] = (target_image, raw_sino, rec, prompts, true_c, sc, rnd, truth_image, range_blurred_image, range_sigma, detector_sigma, psf_model, range_model)
            st.session_state["recon_key"] = recon_key
    # The first item stored in recon_result is the comparison target after
    # positron-range blur; the eighth item is the original activity phantom.
    # Keep these names distinct: the convergence study uses target_image,
    # while truth_image is retained for geometry and the original profile.
    target_image, raw_sino, rec, prompts, true_c, sc, rnd, truth_image, range_blurred_image, used_range_sigma, used_detector_sigma, used_psf_model, used_range_model = st.session_state["recon_result"]
    # Streamlit reruns the script after every widget interaction. Rebuild the
    # deterministic sparse system matrix here: it was previously created only
    # inside the Run reconstruction button branch, causing NameError: A is not defined
    # when iteration/subset comparison widgets triggered a later rerun.
    A = build_projector(truth_image.shape[0], views, 64)
    if used_range_model and used_range_sigma > 1e-6:
        H_recon = build_image_range_operator(truth_image.shape[0], round(float(used_range_sigma), 3))
        A = (A @ H_recon).tocsr()
    if used_psf_model and used_detector_sigma > 1e-6:
        G_recon = build_sinogram_detector_psf_operator(
            views, 64, round(float(used_detector_sigma), 3), truth_image.shape[0])
        A = (G_recon @ A).tocsr()
    # For comparison, scale reconstruction to the mean of the non-zero truth region.
    # Compare against the physically appropriate target (including the
    # positron-range blur), not the unblurred source phantom.
    mask = target_image > 0
    rec_scale = rec * (np.mean(target_image[mask]) / max(np.mean(rec[mask]), 1e-9))
    diff = rec_scale - target_image
    norm_rmse = float(np.sqrt(np.mean((rec_scale[mask] - target_image[mask]) ** 2)) / max(np.mean(target_image[mask]), 1e-9))
    c1, c2 = st.columns(2)
    with c1: st.pyplot(fig_image(truth_image, "Original activity phantom")); plt.close("all")
    with c2: st.pyplot(fig_sinogram(raw_sino, "Noisy measured sinogram")); plt.close("all")
    c3, c4 = st.columns(2)
    with c3: st.pyplot(fig_image(rec, f"Reconstruction: {algorithm}")); plt.close("all")
    with c4: st.pyplot(fig_image(diff, "Difference map (scaled reconstruction − range-blurred target)", cmap="coolwarm")); plt.close("all")
    m1, m2, m3 = st.columns(3)
    m1.metric("Normalised RMSE", f"{norm_rmse:.3f}")
    m2.metric("Prompts", f"{int(prompts.sum()):,}")
    m3.metric("Total subset updates", str(iterations*(n_subsets if algorithm=="OSEM" else 1) if algorithm in ("MLEM","OSEM") else "n/a"))
    if recon_phantom_name == "Two hot sources":
        st.markdown("### Two-source resolution and quantification")
        st.markdown(f"**Phantom definition:** left source = {two_diam_recon:.1f} mm at {two_hot_recon:.1f}× background; right source = {two_diam_right_recon:.1f} mm at {two_hot_right_recon:.1f}× background; centre spacing = {two_sep_recon:.1f} mm. The two ROIs are sized independently.")
        st.markdown("The two hot ROIs use their own physical diameters, and a central gap ROI tests whether activity from the neighbouring sources spills into the space between them. For the paired PSF experiment, the exact same prompt events are reconstructed with and without the detector-response model.")
        two_metrics=quantify_two_sources(rec_scale,target_image,two_sep_recon,two_diam_recon,
                                         two_hot_recon,right_hot_activity=two_hot_right_recon,
                                         right_diameter_mm=two_diam_right_recon)
        qm1,qm2,qm3,qm4,qm5=st.columns(5)
        qm1.metric("Left source mean",f"{two_metrics['left_mean']:.2f} × background")
        qm2.metric("Right source mean",f"{two_metrics['right_mean']:.2f} × background")
        qm3.metric("Gap / hot contrast",f"{two_metrics['valley_to_peak_pct']:.1f}%")
        qm4.metric("Gap activity",f"{two_metrics['gap_mean']:.2f} × background")
        qm5.metric("Separation / FWHM",f"{two_sep_recon/eff_fwhm_no:.2f}")
        roi_fig,roi_ax=plt.subplots(figsize=(5.0,4.5))
        roi_ax.imshow(rec_scale,cmap="magma",origin="lower")
        for roi_mask,color,label in [(two_metrics['left_roi'],'#42d6b0','Left hot ROI'),(two_metrics['right_roi'],'#64b5ff','Right hot ROI'),(two_metrics['gap_roi'],'white','Central gap ROI')]:
            roi_ax.contour(roi_mask.astype(float),levels=[0.5],colors=[color],linewidths=1.3)
        roi_ax.set_title("Reconstructed image with quantitative ROIs");roi_ax.set_xticks([]);roi_ax.set_yticks([]);roi_fig.tight_layout()
        st.pyplot(roi_fig,use_container_width=False);plt.close(roi_fig)
        profile_fig,profile_ax=plt.subplots(figsize=(8,2.6))
        midrow=rec_scale.shape[0]//2
        profile_ax.plot(truth_image[midrow],label="Original activity",lw=1.5,alpha=.7)
        profile_ax.plot(range_blurred_image[midrow],label="After positron-range blur",lw=1.8)
        profile_ax.plot(rec_scale[midrow],label="Reconstructed",lw=1.8)
        profile_ax.axvspan((64-1)/2-two_sep_recon/2-two_diam_recon/2,
                           (64-1)/2-two_sep_recon/2+two_diam_recon/2,
                           color="#42d6b0",alpha=.12,label="Left source extent")
        profile_ax.axvspan((64-1)/2+two_sep_recon/2-two_diam_right_recon/2,
                           (64-1)/2+two_sep_recon/2+two_diam_right_recon/2,
                           color="#64b5ff",alpha=.12,label="Right source extent")
        profile_ax.set_xlabel("Image position (pixels; 1 pixel ≈ 1 mm)");profile_ax.set_ylabel("Relative activity");profile_ax.grid(alpha=.2);profile_ax.legend(ncol=3,fontsize=8)
        profile_fig.tight_layout();st.pyplot(profile_fig,use_container_width=True);plt.close(profile_fig)
        st.caption(f"Left source: {two_diam_recon:.1f} mm, {two_hot_recon:.1f}× background · right source: {two_diam_right_recon:.1f} mm, {two_hot_right_recon:.1f}× background · centre separation: {two_sep_recon:.1f} mm. Source means are background-normalised. Recovery is background-subtracted and compared with the range-blurred target. The gap-to-hot contrast ratio is a simple cross-talk/separability indicator, not a formal clinical lesion-detectability metric.")
        with st.expander("Controlled PSF demonstration: one merged peak before modelling, two peaks after restoration", expanded=True):
            st.markdown("This is a deliberately controlled **image-domain** experiment using the same unequal-size phantom. First, the positron-range-blurred target is convolved with a known detector Gaussian PSF. Richardson–Lucy restoration then models and reverses only that detector blur. It isolates the PSF concept from the projection geometry used in MLEM/OSEM below; it is not a substitute for the paired tomographic reconstruction or a claim that PSF can recover information beyond the physical resolution limit.")
            demo_c1, demo_c2 = st.columns(2)
            with demo_c1:
                demo_detector_fwhm = st.slider("Detector PSF FWHM for this demonstration (mm)", 2.0, 7.0, 4.5, 0.25, key="two_source_demo_fwhm_v14")
            with demo_c2:
                demo_rl_iterations = st.slider("PSF restoration iterations", 1, 20, 8, 1, key="two_source_demo_rl_iters_v14")
            demo_sigma_px = demo_detector_fwhm / 2.355  # 1-mm image pixels
            demo_observed = gaussian_filter(target_image, sigma=demo_sigma_px, mode="reflect")
            demo_restored = richardson_lucy_gaussian(demo_observed, demo_sigma_px, demo_rl_iterations)
            dcols = st.columns(3)
            with dcols[0]:
                st.pyplot(fig_image(target_image, "Target after positron range")); plt.close("all")
            with dcols[1]:
                st.pyplot(fig_image(demo_observed, "Measured · detector PSF included")); plt.close("all")
            with dcols[2]:
                st.pyplot(fig_image(demo_restored, "PSF-aware restored estimate")); plt.close("all")
            demo_off_q = quantify_two_sources(demo_observed, target_image, two_sep_recon, two_diam_recon,
                                               two_hot_recon, right_hot_activity=two_hot_right_recon,
                                               right_diameter_mm=two_diam_right_recon)
            demo_on_q = quantify_two_sources(demo_restored, target_image, two_sep_recon, two_diam_recon,
                                              two_hot_recon, right_hot_activity=two_hot_right_recon,
                                              right_diameter_mm=two_diam_right_recon)
            dmetrics = st.columns(4)
            dmetrics[0].metric("Small-source recovery · before", f"{demo_off_q['left_recovery_vs_target_pct']:.1f}%")
            dmetrics[1].metric("Small-source recovery · after", f"{demo_on_q['left_recovery_vs_target_pct']:.1f}%")
            dmetrics[2].metric("Large-source recovery · before / after", f"{demo_off_q['right_recovery_vs_target_pct']:.1f}% / {demo_on_q['right_recovery_vs_target_pct']:.1f}%")
            dmetrics[3].metric("Gap / hot contrast · before / after", f"{demo_off_q['valley_to_peak_pct']:.1f}% / {demo_on_q['valley_to_peak_pct']:.1f}%")
            demo_fig, demo_ax = plt.subplots(figsize=(8.2, 2.6))
            demo_mid = target_image.shape[0] // 2
            demo_ax.plot(range_blurred_image[demo_mid], label="Target after positron range", lw=1.8)
            demo_ax.plot(demo_observed[demo_mid], label="Detector-blurred", lw=1.8)
            demo_ax.plot(demo_restored[demo_mid], label="PSF restored", lw=1.8)
            demo_ax.set_xlabel("Position (mm; 1 pixel ≈ 1 mm)"); demo_ax.set_ylabel("Relative activity")
            demo_ax.grid(alpha=.2); demo_ax.legend(fontsize=8, ncol=3)
            demo_fig.tight_layout(); st.pyplot(demo_fig, use_container_width=True); plt.close(demo_fig)
            st.info("Read the line profile from left to right: the smaller 3-mm / 4× source is most affected by partial-volume loss; the larger 5-mm / 8× source retains more signal. In this noise-free known-PSF demonstration, restoration can recover two peaks. Real noisy PET restoration is less stable: recovery depends on counts, PSF accuracy, positron range and regularisation, and excessive iterations can amplify noise or create ringing.")
    if algorithm in ("MLEM","OSEM"):
        st.markdown("### Reconstruction convergence explorer: iterations, subsets and image quality")
        st.markdown("Use the same simulated prompt data to compare MLEM (one full-data update) with OSEM using different ordered-subset counts. Each plotted point is measured after one complete pass through the projection dataset. OSEM performs the pass as sequential subset updates; **one subset is the MLEM special case**.")
        cc1,cc2=st.columns(2)
        with cc1:
            study_passes=st.slider("Maximum complete data passes",2,10,6,1,key="conv_passes")
        with cc2:
            study_subsets=st.multiselect("OSEM subset configurations",[1,2,4,6,8,12],default=[1,2,4,8],key="conv_subset_config")
        if not study_subsets:
            study_subsets=[1]
            st.info("Select at least one subset configuration; one subset corresponds to MLEM.")
        st.caption("Metrics: background CV and SNR describe noise; hot-structure contrast recovery and CNR describe lesion conspicuity; normalised RMSE and Poisson log-likelihood describe agreement with the simulated target/data. SNR is not a universal image-quality score, so interpret it alongside recovery and noise.")
        study_key=(recon_key,study_passes,tuple(study_subsets),used_psf_model)
        if st.button("Run MLEM/OSEM convergence study",type="primary",key="run_convergence_study"):
            study_results=[]
            progress=st.progress(0.0,text="Calculating reconstruction histories...")
            configs=sorted(set([1]+[int(v) for v in study_subsets]))
            for idx,subset_n in enumerate(configs):
                hist=reconstruct_history(prompts,A,truth_image.shape,sc,rnd,n_subsets=subset_n,max_iterations=study_passes)
                for item in hist:
                    img=item["image"]
                    # Normalise only for the ROI-based comparisons; retain the raw likelihood.
                    target_bg=(target_image>=0.85)&(target_image<=1.15)
                    if np.count_nonzero(target_bg)<8:
                        target_bg=(target_image>0)&(target_image<=np.quantile(target_image[target_image>0],0.60))
                    recon_bg=(img[target_bg] if np.any(target_bg) else img[img>0])
                    truth_bg=(target_image[target_bg] if np.any(target_bg) else target_image[target_image>0])
                    scale=float(np.mean(truth_bg))/max(float(np.mean(recon_bg)),1e-9) if len(recon_bg) else 1.0
                    scaled=img*scale
                    mets=reconstruction_metrics(scaled,target_image)
                    study_results.append({"Algorithm":"MLEM" if subset_n==1 else f"OSEM · {subset_n} subsets",
                                          "Subsets":subset_n,"Complete data passes":item["iteration"],
                                          "Subset updates":item["subset_updates"],
                                          "Poisson log-likelihood":item["log_likelihood"],**mets})
                progress.progress((idx+1)/len(configs),text=f"Completed subset configuration {subset_n}")
            progress.empty()
            st.session_state["recon_convergence_study"]={"key":study_key,"table":pd.DataFrame(study_results)}
        stored_study=st.session_state.get("recon_convergence_study")
        if stored_study and stored_study.get("key")==study_key:
            study_df=stored_study["table"].copy()
            st.dataframe(study_df.drop(columns=["Subsets"]).round(3),hide_index=True,use_container_width=True)
            # One independent chart per metric makes trends readable and allows different scales.
            chart_metrics=[("Background CV (%)","Lower is generally less noisy"),
                           ("Background SNR","Higher is generally better, for a stable task/ROI"),
                           ("Hot contrast recovery (%)","Closer to 100% is better for this phantom target"),
                           ("Hot-to-background CNR","Higher improves detectability, provided recovery is not biased"),
                           ("Normalised RMSE","Lower is closer to the simulated target"),
                           ("Poisson log-likelihood","Higher is better for the measured-data likelihood")]
            for met_name,help_text in chart_metrics:
                figc,axc=plt.subplots(figsize=(8,2.7))
                for label,group in study_df.groupby("Algorithm",sort=False):
                    group=group.sort_values("Complete data passes")
                    vals=pd.to_numeric(group[met_name],errors="coerce")
                    if vals.notna().any():
                        axc.plot(group["Complete data passes"],vals,marker="o",label=label)
                axc.set_xlabel("Complete projection-data passes (nominal iterations)")
                axc.set_ylabel(met_name);axc.grid(alpha=.25);axc.legend(fontsize=8,ncol=2)
                axc.set_title(help_text,fontsize=9)
                figc.tight_layout();st.pyplot(figc,use_container_width=True);plt.close(figc)
            st.markdown("**How to read the curves**")
            st.markdown("""- If contrast recovery rises while background CV rises, additional iterations are trading lower bias for higher variance.
- If likelihood continues to improve but CNR or task-specific detectability stops improving, the statistical objective is not the same as the clinical/teaching task objective.
- If OSEM with many subsets advances rapidly at first but becomes noisy or oscillatory, reduce subsets or stop earlier.
- Use the earliest stable stopping region that meets the task's recovery and noise needs; there is no universal best iteration/subset pair.""")
            st.caption("Fairness note: a complete OSEM pass processes all projection angles but splits them across several sequential updates. Equal nominal passes are comparable in total projection data considered, but not identical in arithmetic overhead or optimisation trajectory. The log-likelihood is evaluated against the full dataset after each complete pass.")
        else:
            st.info("Run the convergence study to generate the SNR, noise, recovery, CNR, error and likelihood curves from this exact simulated acquisition.")
        with st.expander("Current reconstruction metrics"):
            current_metrics=reconstruction_metrics(rec_scale,target_image)
            q1,q2,q3=st.columns(3)
            q1.metric("Background SNR",f"{current_metrics['Background SNR']:.2f}")
            q2.metric("Hot contrast recovery",f"{current_metrics['Hot contrast recovery (%)']:.1f}%" if np.isfinite(current_metrics['Hot contrast recovery (%)']) else "n/a")
            q3.metric("Hot/background CNR",f"{current_metrics['Hot-to-background CNR']:.2f}" if np.isfinite(current_metrics['Hot-to-background CNR']) else "n/a")
        with st.expander("Compare PSF modelling on/off using identical prompt data"):
            st.markdown("This paired experiment uses identical measured prompts. The selected isotope's range-response modelling setting is held fixed in both arms; only the scanner detector PSF model changes. The detector FWHM slider above sets the width of the common scanner kernel.")
            if st.button("Run paired resolution-model comparison",key="run_psf_pair"):
                A_geom=build_projector(64,views,64)
                if used_range_model and used_range_sigma > 1e-6:
                    H_pair=build_image_range_operator(64,round(float(used_range_sigma),3))
                    A_geom=(A_geom @ H_pair).tocsr()
                G_psf=build_sinogram_detector_psf_operator(
                    views, 64, round(float(used_detector_sigma),3), 64)
                A_with_psf=(G_psf @ A_geom).tocsr()
                pair=[]
                target_bg=(target_image>=0.85)&(target_image<=1.15)
                if np.count_nonzero(target_bg)<8:
                    target_bg=(target_image>0)&(target_image<=np.quantile(target_image[target_image>0],0.60))
                for label,operator in [("Detector PSF OFF",A_geom),("Detector PSF ON",A_with_psf)]:
                    trial=reconstruct(prompts,operator,truth_image.shape,algorithm,iterations,sc,rnd,n_subsets=n_subsets)
                    if post_sigma>0: trial=gaussian_filter(trial,post_sigma)
                    if np.any(target_bg):
                        norm=float(np.mean(target_image[target_bg]))/max(float(np.mean(trial[target_bg])),1e-9)
                    else:
                        norm=1.0
                    trial=trial*norm
                    m_trial=reconstruction_metrics(trial,target_image)
                    pair.append({"Model":label,"image":trial,"metrics":m_trial})
                st.session_state["psf_pair_result"]={"key":recon_key,"pair":pair}
            stored_psf=st.session_state.get("psf_pair_result")
            if stored_psf and stored_psf.get("key")==recon_key:
                pair=stored_psf["pair"]
                pcols=st.columns(2)
                common_vmax=max(float(np.percentile(np.concatenate([it["image"].ravel() for it in pair]),99.5)),1.0)
                for col,item in zip(pcols,pair):
                    with col:
                        st.pyplot(fig_image(item["image"],item["Model"],vmin=0,vmax=common_vmax));plt.close("all")
                        st.metric("Background CV",f"{item['metrics']['Background CV (%)']:.2f}%")
                        st.metric("Background SNR",f"{item['metrics']['Background SNR']:.2f}")
                        st.metric("Hot recovery",f"{item['metrics']['Hot contrast recovery (%)']:.1f}%" if np.isfinite(item['metrics']['Hot contrast recovery (%)']) else "n/a")
                if recon_phantom_name == "Two hot sources":
                    roi_rows=[]
                    for item in pair:
                        two_q=quantify_two_sources(item["image"],target_image,two_sep_recon,two_diam_recon,
                                                   two_hot_recon,right_hot_activity=two_hot_right_recon,
                                                   right_diameter_mm=two_diam_right_recon)
                        roi_rows.append({"Reconstruction":item["Model"],
                                         f"Left ROI ({two_diam_recon:.1f}-mm source; × background)":two_q["left_mean"],
                                         f"Right ROI ({two_diam_right_recon:.1f}-mm source; × background)":two_q["right_mean"],
                                         "Left recovery vs range-blurred target (%)":two_q["left_recovery_vs_target_pct"],
                                         "Right recovery vs range-blurred target (%)":two_q["right_recovery_vs_target_pct"],
                                         "Gap / mean hot (%)":two_q["valley_to_peak_pct"]})
                    st.markdown("**Two-source ROI quantification: PSF model OFF versus ON**")
                    st.dataframe(pd.DataFrame(roi_rows).round(2),hide_index=True,use_container_width=True)
                linefig,lineax=plt.subplots(figsize=(8,2.5))
                midline=target_image.shape[0]//2
                lineax.plot(range_blurred_image[midline],label="After positron-range blur",lw=2)
                lineax.plot(target_image[midline],label="Selected reconstruction target",lw=1.5,ls=":")
                lineax.plot(pair[0]["image"][midline],label="PSF model OFF")
                lineax.plot(pair[1]["image"][midline],label="PSF model ON")
                lineax.set_xlabel("Pixel position");lineax.set_ylabel("Relative activity");lineax.grid(alpha=.2);lineax.legend(fontsize=8)
                linefig.tight_layout();st.pyplot(linefig,use_container_width=True);plt.close(linefig)
                st.caption("Both images use the same display scale and prompt events. The scanner detector kernel is common to all radionuclides; its FWHM is the selected scanner setting. The isotope-specific positron-range kernel is a separate response and is included in both arms only when its model checkbox is selected. Detector PSF modelling can improve recovery but may amplify noise or produce edge overshoot; it cannot remove positron range unless that separate range response is also modelled.")
            else:
                st.caption("Run the paired experiment to compare detector-PSF modelling on/off using the exact same prompt data and selected isotope-range modelling setting.")
    st.markdown("### Central image profile")
    mid = truth_image.shape[0] // 2
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(truth_image[mid], label="Original phantom", lw=1.5, alpha=0.8)
    ax.plot(range_blurred_image[mid], label="After positron-range blur", lw=2)
    ax.plot(target_image[mid], label="Selected reconstruction target", lw=1.5, ls=":")
    ax.plot(rec_scale[mid], label="Reconstruction", lw=1.5)
    ax.set_xlabel("Pixel position"); ax.set_ylabel("Relative activity"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with st.expander("Extended learning: how PET reconstruction works"):
        st.markdown("**Reconstruction is an inverse problem.** The scanner measures noisy projection data; the image is not measured directly. A system matrix A describes how each image voxel contributes to each measurement bin. In a simplified linear model, the expected measurement is Ax, with additive estimates for scatter and random coincidences. Real systems use more complete models of detector geometry, attenuation, normalisation, TOF and resolution.")
        st.latex(r"y_i\sim\operatorname{Poisson}\left(\sum_j a_{ij}x_j+s_i+r_i\right)")
        st.markdown("**Backprojection** smears each measurement back along its ray. Simple backprojection therefore creates a characteristic blur. **Filtered backprojection (FBP)** applies a frequency-domain ramp filter to counteract the blur before backprojection. The filter restores high spatial frequencies but can also amplify noise; practical systems often apply a windowed filter.")
        st.markdown("**MLEM** maximises a Poisson likelihood under the assumed forward model. Each full update uses all measured bins. **OSEM** divides the data into subsets and applies sequential partial EM updates. It is a computational acceleration technique: one subset makes it the MLEM update, while multiple subsets produce faster early progress but generally do not guarantee exact convergence to the same ML solution. Subset ordering, number of subsets, iterations and post-filtering affect the result.")
        st.markdown("**Bias–variance trade-off:** early iterations can improve recovery of small/hot structures; later iterations may amplify noise and artefacts. Post-filtering reduces noise at the cost of resolution and contrast. A fair comparison should report iterations and subsets (or total subset updates), filter, voxel size, corrections, counts, and task-specific metrics—not just the algorithm name.\n\n**Key MLEM parameters:** number of full-data iterations / stopping point; initial image; attenuation, scatter, randoms and normalisation model; system matrix / resolution model; voxel size; any post-filter or regularisation.\n\n**Key OSEM parameters:** number of iterations; number of subsets; subset ordering/balancing; total subset updates (iterations × subsets); system matrix / PSF / TOF modelling; corrections; voxel size; post-filtering and stopping point. More subsets are not automatically better: subsets that are too numerous can reduce per-subset information and produce noisy or limit-cycle behaviour.\n\n**Key FBP parameters:** projection sampling; filter type (ramp, Hann, Hamming, Butterworth); filter cut-off / window; pixel size; any prefiltering and correction strategy.\n\n**PSF modelling:** the scanner detector/system response is incorporated in the forward model during iterative reconstruction. Its FWHM is a scanner property and is not changed simply by selecting another radionuclide. The positron-range response is a separate, isotope-specific kernel. This application lets students model these two components independently: detector PSF FWHM is user-set; the positron-range kernel is selected from the isotope table. Their convolution gives an isotope-dependent effective response. Modelling can improve apparent resolution and small-object recovery, but may increase noise and produce edge overshoot/Gibbs artefacts. It cannot restore information beyond finite counts, sampling or an inaccurate kernel.")
        st.latex(r"\hat x=\arg\max_{x\geq0}\;\sum_i\left[y_i\ln(Ax+s+r)_i-(Ax+s+r)_i\right]")
    if recon_phantom_name == "Two hot sources":
        with st.expander("Teaching notes: resolving and quantifying two nearby hot sources"):
            st.markdown("**Why this experiment matters:** when two hot structures are close relative to the system FWHM, their activity distributions overlap after positron transport and detector response. In the reconstructed image, the two peaks can merge into one broad maximum. The central valley becomes shallower, and activity spills between the source ROIs. This is both a resolution problem and a quantification problem.")
            st.markdown("**What PSF modelling changes:** the acquired sinogram still contains the physical detector blur. PSF-aware MLEM/OSEM includes the detector response in the system matrix, allowing the algorithm to account for that known blur when estimating the source image. Compare the two ROI means, the central gap-to-hot ratio and the line profile. Better separation may come with higher noise or edge overshoot; PSF modelling does not undo the positron-range component.")
            st.markdown("**How to run the exercise:** start with a separation larger than the detector FWHM; reduce the separation in 0.5 mm steps; compare PSF model OFF and ON on the same prompts; then repeat with a larger detector FWHM or a higher-range isotope. Students should report source diameter, centre spacing, detector FWHM, isotope, counts, iterations/subsets and the two source means—not just whether the image looks sharper.")
            st.latex(r"RC_{ROI}=100\times\frac{\bar C_{recon,ROI}-\bar C_{bg}}{\bar C_{target,ROI}-\bar C_{bg}}")
            st.latex(r"V/P(\%)=100\times\frac{\bar C_{gap}}{(\bar C_{left}+\bar C_{right})/2}")
            st.caption("The gap-to-peak ratio is an illustrative cross-talk/separability measure for this digital phantom. It is not a standardised resolution or lesion-detectability metric.")
    with st.expander("What each algorithm teaches"):
        st.markdown("- **Simple backprojection:** distributes each projection measurement back along its ray; without filtering it produces a blurred image.\n- **FBP:** filters projection data (here with a ramp filter) before backprojection.\n- **MLEM:** maximum-likelihood expectation maximisation. Each full iteration uses every projection bin to make one statistically motivated image update. It is comparatively computationally demanding.\n- **OSEM:** ordered-subsets expectation maximisation. It partitions projection angles into subsets and performs several smaller EM updates during one nominal iteration, usually accelerating early convergence. **MLEM is the special case of OSEM with one subset.** With multiple subsets, OSEM is not guaranteed to converge exactly to the MLEM maximum-likelihood solution; it can show limit-cycle behaviour or noise amplification, so iterations × subsets and stopping point matter.\n\n**Fair comparison:** one OSEM iteration with 6 subsets performs 6 subset updates, whereas one MLEM iteration performs one full-data update. Do not compare iteration numbers alone; compare total subset updates, computation time, contrast recovery and noise.")
        st.latex(r"x_j^{(k+1)}=\frac{x_j^{(k)}}{\sum_i a_{ij}}\sum_i a_{ij}\frac{y_i}{\sum_l a_{il}x_l^{(k)}+s_i+r_i}")
        st.latex(r"\text{OSEM subset }b:\quad x_j^{(k,b+1)}=\frac{x_j^{(k,b)}}{\sum_{i\in S_b}a_{ij}}\sum_{i\in S_b}a_{ij}\frac{y_i}{\sum_l a_{il}x_l^{(k,b)}+s_i+r_i}")
        st.dataframe(pd.DataFrame([
            ["Projection use", "All projection bins in each update", "One subset of projection angles per sub-update"],
            ["Speed", "Slower per full update", "Faster early updates in typical use"],
            ["Subset count = 1", "Same full-data EM update", "Exactly MLEM in this implementation"],
            ["Convergence", "Approaches the ML solution under model assumptions", "Multiple subsets can oscillate / limit-cycle and need a stopping rule"],
        ],columns=["Comparison", "MLEM", "OSEM"]),hide_index=True,use_container_width=True)
    with st.expander("PSF, TOF and partial-volume effects (PVE): how they fit together"):
        st.markdown("**PSF / resolution modelling** incorporates the detector/system response in the iterative forward model. In this application the PSF is represented by a stationary Gaussian response across detector-offset bins within each projection; real PET PSFs can vary with position, direction, depth of interaction and scanner geometry. The demonstrator does not model vendor-specific PSF behaviour or Gibbs artefacts in full.\n\n**TOF (time-of-flight)** uses the measured arrival-time difference to weight the annihilation location along each LOR. Its localisation FWHM is approximately c × timing-resolution / 2. TOF can improve noise properties and convergence, especially in larger objects, but does not simply sharpen the detector PSF. Full TOF reconstruction requires time bins and a TOF system matrix; this 2-D teaching code currently illustrates the timing localisation but does not implement TOF bins in MLEM/OSEM.\n\n**PVE (partial-volume effect)** is the loss of apparent activity concentration in small structures because the reconstructed system resolution is finite and activity spills across ROI boundaries. It is task- and size-dependent. PSF modelling can improve recovery, but does not eliminate PVE universally.\n\nThe selected detector PSF FWHM is a user-set scanner parameter; it should be measured or specified for the actual system. The term “POV” is not a standard PET reconstruction acronym. The likely related terms are **TOF** (time-of-flight) or **PVE** (partial-volume effect).")
        st.latex(r"\Delta x_{TOF,FWHM}=\frac{c\,\Delta t_{FWHM}}{2}")
        st.latex(r"FWHM_{eff}\approx\sqrt{FWHM_{range}^2+FWHM_{detector}^2+FWHM_{acollinearity}^2}")
    st.markdown("### MLEM and OSEM share the same statistical foundation")
    st.markdown("Both methods start from a Poisson likelihood model for measured PET projection data. MLEM performs a full-data expectation–maximisation update. OSEM partitions the projection bins into ordered subsets and applies sequential partial EM updates. OSEM is therefore an acceleration strategy built on the same EM framework, not a different physical measurement model. With one subset, its update is MLEM; with multiple subsets, early progress is faster but exact convergence to the full-data maximum-likelihood solution is not guaranteed.")
    st.latex(r"y_i\sim\operatorname{Poisson}\!\left(\sum_j a_{ij}x_j+s_i+r_i\right)")
    shared_cols=st.columns(3)
    shared_cols[0].markdown("**Shared root**\n\nPoisson likelihood + EM")
    shared_cols[1].markdown("**MLEM**\n\nAll projection bins → one full-data update")
    shared_cols[2].markdown("**OSEM**\n\nOrdered subsets → sequential partial updates")
    st.caption("Special case: OSEM with one subset uses the full dataset per update and is equivalent to MLEM in this implementation.")

    with st.expander("Further reading: reconstruction, PSF, TOF and PVE"):
        st.markdown("- [EANM PET reconstruction technical guide — reconstruction updates and resolution modelling](https://eanm.org/wp-content/uploads/2024/06/EANM22_TechGuide.pdf)\n- [Resolution modelling in PET: theory, practice, benefits and pitfalls](https://pmc.ncbi.nlm.nih.gov/articles/PMC3663852/)\n- [Time-of-flight in perspective: instrumental and computational aspects](https://pmc.ncbi.nlm.nih.gov/articles/PMC8454900/)\n- [Effect of varying OSEM subsets on PET lesion detectability](https://pmc.ncbi.nlm.nih.gov/articles/PMC3856855/)\n- [PET quantification and partial-volume effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC5541089/)")
    st.warning("Model limitations: 2-D parallel-beam geometry; isotope-specific positron-range blur is a Gaussian-equivalent teaching approximation; detector PSF is a stationary Gaussian system matrix; scatter/randoms are simplified. TOF localisation is illustrated but TOF bins are not included in the reconstruction operator. Not a clinical reconstruction engine.")

# -----------------------------------------------------------------------------
# QC laboratory
# -----------------------------------------------------------------------------
with modules[5]:
    st.subheader("PET phantom and quality-control laboratory")
    st.write("Choose a recognised phantom design, generate a synthetic acquisition, display protocol-shaped ROIs and calculate the corresponding image-quality metrics.")
    phantom_type = st.selectbox("QC phantom", [
        "Preclinical PET — NEMA NU 4-2008 image-quality phantom",
        "Clinical PET — NEMA/IEC NU 2 image-quality phantom",
        "Clinical PET — ACR accreditation phantom (SUV ROI exercise)",
        "Uniform cylindrical phantom — calibration / constancy"
    ])
    resolution = st.slider("Simulated reconstructed FWHM (mm)", 0.5 if phantom_type.startswith("Preclinical") else 1.0, 10.0 if phantom_type.startswith("Clinical") else 4.0,
                           1.0 if phantom_type.startswith("Preclinical") else 5.0, 0.25 if phantom_type.startswith("Preclinical") else 0.5)
    count_level = st.select_slider("NU 4 simulated count scale", options=[300, 600, 1200, 1800, 3000, 6000], value=1800) if phantom_type.startswith("Preclinical") else 1800
    qc_psf_model = st.checkbox("Apply detector-PSF restoration (RL demo)", value=False, key="qc_psf_model")
    isotope_range_fwhm = p["range_fwhm_mm"]
    st.caption(f"Selected isotope: {isotope} · mean positron range in water {p['mean_range_mm']:.1f} mm · representative projected range-kernel FWHM {isotope_range_fwhm:.2f} mm. This is not the scanner detector PSF; the scanner resolution is set separately by the QC reconstructed-FWHM control.")
    noise_pct = st.slider("Additional image noise (%)", 0.0, 15.0, 2.0, 0.5)
    qc_duration=st.slider("Phantom acquisition duration (min)",1,60,10,1,key="qc_duration")
    qc_decay_factor=decay_average_activity_factor(p["half_life_min"],qc_duration)
    st.caption(f"At {qc_duration} min, mean activity is {100*qc_decay_factor:.1f}% of the starting activity; this and β⁺ branching influence the simulated counting statistics.")
    fault = st.selectbox("Optional teaching fault", ["None", "Calibration scale +20%", "Radial sensitivity loss", "Reduced counts", "Extra blur", "Cold-chamber spillover (+5% background)"], key="qc_phantom_fault")
    seed = st.slider("Noise seed", 1, 100, 13, key="qc_phantom_seed")

    if phantom_type.startswith("Preclinical"):
        ideal_vol, acquired_vol, geom = build_nu4_volume(
            counts_per_unit=float(count_level), resolution_fwhm_mm=resolution,
            seed=seed, isotope_range_fwhm_mm=isotope_range_fwhm,
            psf_model=qc_psf_model, positron_branch=p["beta_plus_branch"],
            decay_factor=qc_decay_factor, acquisition_duration_min=qc_duration
        )
        if noise_pct > 0:
            rng=np.random.default_rng(seed+101)
            acquired_vol=np.maximum(acquired_vol+rng.normal(0, noise_pct/100.0, acquired_vol.shape),0)
        if fault == "Calibration scale +20%": acquired_vol *= 1.20
        elif fault == "Radial sensitivity loss":
            radial=np.sqrt(geom["X"]**2+geom["Y"]**2)
            acquired_vol *= np.clip(1-0.30*radial/15.0,0.65,1.0)[None,:,:]
        elif fault == "Reduced counts":
            rng=np.random.default_rng(seed+202)
            acquired_vol=rng.poisson(np.maximum(acquired_vol*0.15,0))/0.15
        elif fault == "Extra blur": acquired_vol=gaussian_filter(acquired_vol, sigma=(0,1.5,1.5))
        elif fault == "Cold-chamber spillover (+5% background)":
            for cx,cy,_ in geom["cold_centres"]:
                chamber=(geom["X"]-cx)**2+(geom["Y"]-cy)**2 <= 2.0**2
                for zi in np.where((geom["z"]>=3.75)&(geom["z"]<11.25))[0]:
                    acquired_vol[zi][chamber]=0.05
        metrics=nu4_roi_metrics(acquired_vol,geom)
        ideal_metrics=nu4_roi_metrics(ideal_vol,geom)
        # Display representative transaxial slices: uniform, rod and spillover regions.
        z=geom["z"]
        iz_u=int(np.argmin(abs(z-44.5))); iz_r=int(np.argmin(abs(z-25.0))); iz_s=int(np.argmin(abs(z-7.5)))
        xy=geom["xy"]; X,Y=geom["X"],geom["Y"]
        fig,axs=plt.subplots(1,3,figsize=(12,3.8))
        axs[0].imshow(acquired_vol[iz_u],origin="lower",extent=[xy[0],xy[-1],xy[0],xy[-1]],cmap="inferno")
        axs[0].add_patch(plt.Circle((0,0),11.25,fill=False,edgecolor="cyan",linewidth=1.5))
        axs[0].set_title("Uniform segment · 22.5-mm ROI")
        axs[1].imshow(acquired_vol[iz_r],origin="lower",extent=[xy[0],xy[-1],xy[0],xy[-1]],cmap="inferno")
        for cx,cy,d in geom["rod_centres"]:
            axs[1].add_patch(plt.Circle((cx,cy),d,fill=False,edgecolor="cyan",linewidth=1.0))
        axs[1].set_title("Hot-rod segment · 1–5 mm rods")
        axs[2].imshow(acquired_vol[iz_s],origin="lower",extent=[xy[0],xy[-1],xy[0],xy[-1]],cmap="inferno")
        for cx,cy,label in geom["cold_centres"]:
            axs[2].add_patch(plt.Circle((cx,cy),2,fill=False,edgecolor="cyan",linewidth=1.4))
            axs[2].text(cx,cy+2.7,label,ha="center",color="cyan",fontsize=8)
        axs[2].set_title("Spillover segment · 4-mm ROIs")
        for ax in axs:
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Uniform VOI mean",f"{metrics['uniform_mean']:.3f}")
        c2.metric("Uniform %SD",f"{metrics['uniform_percent_sd']:.2f}%")
        c3.metric("Uniform Dvar",f"{metrics['uniform_dvar']:.2f}%")
        c4.metric("Uniform min / max",f"{metrics['uniform_min']:.2f} / {metrics['uniform_max']:.2f}")
        st.markdown("### NEMA NU 4-style rod recovery coefficients")
        st.dataframe(metrics["rods"].round(2),hide_index=True,use_container_width=True)
        st.markdown("### Spillover ratio")
        st.dataframe(metrics["spill"].round(3),hide_index=True,use_container_width=True)
        st.markdown("### ROI definitions and equations")
        st.latex(r"\%SD_{uniform}=100\frac{SD_{uniform\ VOI}}{Mean_{uniform\ VOI}}")
        st.latex(r"D_{var}(\%)=100\frac{Maximum-Minimum}{Mean}")
        st.latex(r"RC_d(\%)=100\frac{Mean\ axial\ rod\ profile_d}{Mean_{uniform\ VOI}}")
        st.latex(r"SOR_{air/water}=\frac{Mean_{cold\ chamber}}{Mean_{uniform\ VOI}}")
        st.markdown("**Protocol-shaped VOIs used here:** uniform region: 22.5-mm diameter × 10-mm axial length, centred in the uniform section; rods: central 10-mm rod length with circular ROIs of diameter twice the physical rod diameter; spillover: 4-mm diameter × 7.5-mm axial length in each air/water compartment. The 3-D digital phantom uses the nominal 50-mm length, 15-mm radius and 15/20/15-mm axial regions.")
        st.warning("This is a synthetic NU 4-style training calculation. It approximates the published ROI geometry and metrics, but it is not a formal NEMA test implementation: actual NU 4 acquisition, reconstruction, VOI placement, rod profiles, scanner corrections and reporting must follow the licensed standard and the scanner-specific protocol. NEMA does not define universal pass limits for these metrics.")
        with st.expander("Teaching note: what does each NU 4 metric tell us?"):
            st.markdown("- **Uniform %SD / COV:** image noise and uniformity in the central uniform section. Lower values generally indicate less variation, but depend strongly on counts and reconstruction.")
            st.markdown("- **Dvar:** relative max-to-min spread in the uniform VOI; sensitive to nonuniformity and noise extremes.")
            st.markdown("- **Rod RC:** partial-volume recovery as a function of rod diameter. Smaller rods are more affected by finite spatial resolution.")
            st.markdown("- **Air/water SOR:** activity measured in non-emitting compartments relative to the uniform region; non-zero spillover reflects scatter, penetration, reconstruction and correction effects.")
        st.markdown("**Assessment:** compare these results with the scanner's commissioning baseline, manufacturer specification and locally approved action limits. Record reconstruction algorithm, iterations/subsets, filter, corrections, voxel size, activity concentration and scan duration. Do not invent a pass/fail threshold from a single simulated run.")
        st.markdown("Reference: [NEMA NU4-2008 image-quality phantom geometry and ROI analysis (Bahri et al., JNM 2009)](https://jnm.snmjournals.org/content/50/10/1730.full).")

    elif phantom_type.startswith("Clinical PET — NEMA"):
        ratio_choice=st.selectbox("Hot-sphere to background activity concentration ratio",[4.0,8.0],index=0,format_func=lambda v:f"{v:.0f}:1")
        acquired_stack,geom=build_nu2_iec_slice_stack(sphere_background_ratio=ratio_choice,
            resolution_fwhm_mm=resolution,noise_pct=noise_pct,seed=seed,
            isotope_range_fwhm_mm=isotope_range_fwhm,psf_model=qc_psf_model,
            positron_branch=p["beta_plus_branch"],decay_factor=qc_decay_factor,
            acquisition_duration_min=qc_duration)
        if fault == "Calibration scale +20%": acquired_stack *= 1.20
        elif fault == "Radial sensitivity loss":
            rr=np.sqrt(geom["X"]**2+geom["Y"]**2)
            acquired_stack *= np.clip(1-0.15*rr/135.0,0.8,1.0)[None,:,:]
        elif fault == "Reduced counts":
            rng=np.random.default_rng(seed+202)
            acquired_stack=np.maximum(rng.poisson(np.maximum(acquired_stack*20,0))/20,0)
        elif fault == "Extra blur": acquired_stack=gaussian_filter(acquired_stack,sigma=(0,1.5,1.5))
        sphere_table,bg_table,lung_mean,lung_error=nu2_iec_metrics(acquired_stack,geom)
        X,Y=geom["X"],geom["Y"]
        fig,ax=plt.subplots(figsize=(7.0,5.2))
        ax.imshow(acquired_stack[2],origin="lower",extent=[X.min(),X.max(),Y.min(),Y.max()],cmap="inferno")
        for d,(cx,cy) in zip(geom["sphere_ds"],geom["centres"]):
            edge = "#5ee6a8" if d <= 22 else "#55c9f2"
            ax.add_patch(plt.Circle((cx,cy),d/2,fill=False,edgecolor=edge,linewidth=1.8))
            ax.text(cx,cy+d/2+3,f"{d} mm",ha="center",color="white",fontsize=8,
                    bbox=dict(facecolor="#17212b",alpha=.65,edgecolor="none",pad=1))
        ax.add_patch(plt.Circle((0,0),25,fill=False,edgecolor="#f6d365",linewidth=1.7,linestyle="--"))
        ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)");ax.set_title("NEMA/IEC NU 2 · schematic central transaxial slice")
        ax.set_aspect("equal")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],color="#5ee6a8",lw=2,label="Hot spheres (10–22 mm)"),
                           Line2D([0],[0],color="#55c9f2",lw=2,label="Cold spheres (28, 37 mm)"),
                           Line2D([0],[0],color="#f6d365",lw=2,ls="--",label="Central background ROI")],
                  loc="upper right",fontsize=7,framealpha=.85)
        fig.tight_layout();st.pyplot(fig,use_container_width=True);plt.close(fig)
        st.caption("ROI key: green = hot-sphere regions; blue = cold-sphere regions; yellow dashed = central background sampling ROI. Background-variability ROIs are sampled separately over the simulated slices and tabulated below.")
        st.markdown("### Hot/cold sphere contrast recovery")
        st.dataframe(sphere_table.round(2),hide_index=True,use_container_width=True)
        st.markdown("### Background variability by ROI diameter")
        st.dataframe(bg_table.round(2),hide_index=True,use_container_width=True)
        a,b=st.columns(2);a.metric("Lung insert mean",f"{lung_mean:.3f}");b.metric("Lung insert relative error",f"{lung_error:+.1f}%")
        st.latex(r"CRC_{hot,j}(\%)=100\frac{(C_{H,j}/C_B)-1}{(A_H/A_B)-1}")
        st.latex(r"BV_j(\%)=100\frac{SD(\bar C_{B,j,k})}{Mean(\bar C_{B,j,k})}")
        st.markdown("In this teaching example, the hot spheres are 10, 13, 17 and 22 mm, while 28 and 37 mm are cold. The low-density lung insert is represented centrally. The activity ratio can be set to 4:1 or 8:1. The background-variability illustration samples 12 fixed background ROIs across five axial planes (60 measurements per ROI diameter).")
        st.warning("The sphere and lung positions and background ROI grid are schematic. This is not a compliant NU 2 performance test or accreditation analysis; formal work must follow the current licensed NEMA NU 2 standard and specified phantom geometry, acquisition and analysis procedures.")
        with st.expander("Teaching note: interpreting NU 2 image-quality metrics"):
            st.markdown("- **CRC:** fraction of true hot-sphere contrast recovered; smaller spheres usually have lower CRC because of partial-volume effects.\n- **Background variability:** variation across background ROIs and a practical noise indicator.\n- **Lung insert error:** demonstrates the effect of low-density material and correction methods.\n- **Reconstruction dependence:** iterations/subsets and post-filtering trade contrast recovery against background noise.")
        st.markdown("References: [NEMA NU 2-2018 system performance example](https://pmc.ncbi.nlm.nih.gov/articles/PMC7316913/); [NU 2 sphere CRC and background variability definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC13105374/).")

    elif phantom_type.startswith("Clinical PET — ACR"):
        ideal_img, acquired_img, geom=build_acr_pet_slice(resolution_fwhm_mm=resolution,
            noise_pct=noise_pct,seed=seed,isotope_range_fwhm_mm=isotope_range_fwhm,
            psf_model=qc_psf_model,positron_branch=p["beta_plus_branch"],
            decay_factor=qc_decay_factor,acquisition_duration_min=qc_duration)
        if fault == "Calibration scale +20%": acquired_img *= 1.20
        elif fault == "Radial sensitivity loss":
            rr=np.sqrt(geom["X"]**2+geom["Y"]**2)
            acquired_img *= np.clip(1-0.15*rr/108.0,0.8,1.0)
        elif fault == "Reduced counts":
            rng=np.random.default_rng(seed+202)
            acquired_img=np.maximum(rng.poisson(np.maximum(acquired_img*20,0))/20,0)
        elif fault == "Extra blur": acquired_img=gaussian_filter(acquired_img, sigma=2.0)
        metrics=acr_roi_metrics(acquired_img,geom)
        X,Y=geom["X"],geom["Y"]
        fig,ax=plt.subplots(figsize=(6.3,6.1))
        ax.imshow(acquired_img,origin="lower",extent=[-110,110,-110,110],cmap="inferno")
        ax.add_patch(plt.Circle((0,0),32.5,fill=False,edgecolor="#58d7ef",linewidth=1.8,label="65-mm background ROI"))
        for cx,cy,d,label in geom["hot_specs"]:
            ax.add_patch(plt.Circle((cx,cy),12.5,fill=False,edgecolor="#5ee6a8",linewidth=1.5))
            ax.text(cx,cy+16,label,ha="center",color="white",fontsize=8,
                    bbox=dict(facecolor="#17212b",alpha=.65,edgecolor="none",pad=1))
        for cx,cy,label in geom["cold_specs"]:
            ax.add_patch(plt.Circle((cx,cy),12.5,fill=False,edgecolor="#f6c85f",linewidth=1.5))
            ax.text(cx,cy+16,label,ha="center",color="white",fontsize=8,
                    bbox=dict(facecolor="#17212b",alpha=.65,edgecolor="none",pad=1))
        ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)");ax.set_title("ACR PET phantom — schematic transaxial slice")
        ax.set_aspect("equal")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],color="#58d7ef",lw=2,label="Background ROI"),
                           Line2D([0],[0],color="#5ee6a8",lw=2,label="Hot-cylinder fixed ROI"),
                           Line2D([0],[0],color="#f6c85f",lw=2,label="Cold-cylinder fixed ROI")],
                  loc="upper right",fontsize=7,framealpha=.85)
        fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.caption("ROI key: cyan = central background ROI; green = fixed 25-mm-diameter hot-cylinder ROI; yellow = same-size ROI copied onto cold cylinders. The overlay is schematic, not an accreditation positioning template.")
        a,b,c=st.columns(3)
        a.metric("Mean background SUV",f"{metrics['background_mean']:.3f}")
        b.metric("Maximum SUV · 25-mm",f"{metrics['max25']:.3f}")
        c.metric("Maximum SUV ratio · 16/25",f"{metrics['ratio16_25']:.3f}")
        st.markdown("### ROI measurements (same transaxial slice)")
        st.dataframe(metrics["table"].round(3),hide_index=True,use_container_width=True)
        st.markdown("**ROI placement:** one slice through the middle portion of the hot cylinders; a centred circular background ROI 60–70 mm in diameter, avoiding all cylinders; then a 25-mm-diameter ROI on the 25-mm hot cylinder and copied without resizing to each visible hot and cold/Teflon cylinder. For smaller hot cylinders, the copied ROI intentionally includes surrounding background; use the ROI maximum SUV for the hot cylinders.")
        st.latex(r"SUV_{mean,bg}=\frac{1}{N_{bg}}\sum_{i\in ROI_{bg}}SUV_i")
        st.latex(r"R_{16/25}=\frac{SUV_{max,16\,mm}}{SUV_{max,25\,mm}}")
        st.markdown("### ACR SUV criteria — protocol-specific assessment")
        for criterion,passed in metrics["criteria"].items():
            st.write(("✅ PASS — " if passed else "❌ OUTSIDE CRITERION — ")+criterion)
        st.caption("These criteria are only meaningful when the ACR-approved phantom, prescribed activity dilution, SUV calibration, acquisition timing, clinical reconstruction and ROI procedure are followed. Current ACR phantom instructions: mean background SUV 0.90–1.10; maximum SUV in 25-mm cylinder >1.87 and <2.91; maximum SUV 16-mm/25-mm ratio >0.7.")
        st.warning("The displayed phantom component locations are schematic for teaching. They are not a dimensional template of the ACR phantom. This simulation cannot establish ACR accreditation or replace the ACR-approved phantom/protocol.")
        with st.expander("Teaching note: why the ACR ROIs are different from NU 4 ROIs"):
            st.markdown("The ACR method uses a single transaxial slice, a 60–70 mm central background ROI, and a fixed 25-mm ROI copied onto the hot and cold cylinders. The NU 4 method is a preclinical 3-D image-quality assessment using a 22.5-mm × 10-mm uniform VOI, rod-profile recovery coefficients and 4-mm × 7.5-mm cold-compartment VOIs. These ROI methods are not interchangeable.")
        st.markdown("Reference: [ACR PET phantom testing instructions, revised 5 November 2026](https://accreditationsupport.acr.org/support/solutions/articles/11000062800-phantom-testing-pet-revised-11-5-2026).")

    else:
        ideal_img, acquired_img, geom=build_uniform_cylinder(noise_pct=noise_pct,
            resolution_fwhm_mm=resolution,seed=seed,
            isotope_range_fwhm_mm=isotope_range_fwhm,psf_model=qc_psf_model,
            positron_branch=p["beta_plus_branch"],decay_factor=qc_decay_factor,
            acquisition_duration_min=qc_duration)
        if fault == "Calibration scale +20%": acquired_img *= 1.20
        elif fault == "Radial sensitivity loss":
            rr=np.sqrt(geom["X"]**2+geom["Y"]**2)
            acquired_img *= np.clip(1-0.20*rr/30.0,0.75,1.0)
        elif fault == "Reduced counts":
            rng=np.random.default_rng(seed+202)
            acquired_img=np.maximum(rng.poisson(np.maximum(acquired_img*25,0))/25,0)
        elif fault == "Extra blur": acquired_img=gaussian_filter(acquired_img, sigma=2.0)
        mask=geom["mask"] & (geom["X"]**2+geom["Y"]**2 <= 20.0**2)
        vals=acquired_img[mask]; ref=ideal_img[mask]
        mean=float(np.mean(vals)); sd=float(np.std(vals,ddof=1)); cv=100*sd/max(mean,1e-12)
        bias=100*(mean-float(np.mean(ref)))/max(float(np.mean(ref)),1e-12)
        fig,ax=plt.subplots(figsize=(5.5,5));ax.imshow(acquired_img,origin="lower",extent=[geom["X"].min(),geom["X"].max(),geom["Y"].min(),geom["Y"].max()],cmap="inferno")
        ax.add_patch(plt.Circle((0,0),20,fill=False,color="cyan",linewidth=1.5));ax.set_title("Uniform-cylinder constancy image · 40-mm ROI");ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)");ax.set_aspect("equal");fig.tight_layout();st.pyplot(fig);plt.close(fig)
        a,b,c=st.columns(3);a.metric("Mean (relative units)",f"{mean:.3f}");b.metric("CV",f"{cv:.2f}%");c.metric("Bias vs nominal",f"{bias:+.2f}%")
        st.latex(r"CV(\%)=100\frac{SD_{ROI}}{Mean_{ROI}}")
        st.latex(r"Bias(\%)=100\frac{Mean_{measured}-Mean_{reference}}{Mean_{reference}}")
        st.info("Use this uniform cylinder for constancy, uniformity and cross-calibration teaching. A formal quantitative cross-calibration requires known activity concentration, decay correction, traceable dose-calibrator measurement, acquisition time, residual activity and system-specific ROI/VOI protocol.")

    with st.expander("Extended learning: what a PET QC programme should demonstrate"):
        st.markdown("**Acceptance testing / commissioning** establishes documented baseline performance after installation or major service. **Routine constancy testing** checks whether the scanner continues to behave consistently with that baseline. **Performance characterisation** uses standardised methods such as NEMA NU 4 (preclinical) or NU 2 (clinical) to measure defined scanner properties. These purposes are related but are not interchangeable.")
        st.markdown("**Why phantom ROIs matter:** an ROI must be placed in the correct physical compartment, have the prescribed shape and dimensions, use the required axial extent or slice position, and avoid boundary/wall effects. The ROI method determines the metric. A generic circular ROI cannot be substituted for a standard-defined VOI simply because it looks plausible on one image.")
        st.markdown("**Interpretation:** uniform-region variability is primarily a noise/uniformity indicator; rod or sphere recovery probes resolution and partial-volume effects; spillover tests unwanted counts in nominally cold compartments; background variability measures spatial/statistical fluctuation; quantitative bias and cross-calibration assess activity concentration accuracy. These measures should be trended with acquisition/reconstruction metadata held constant.")
        st.latex(r"RC(\%)=100\times\frac{C_{\mathrm{measured}}}{C_{\mathrm{true}}}\quad\text{(simple recovery ratio; protocol-specific definitions may differ)}")
        st.latex(r"SOR=\frac{C_{\mathrm{cold}}}{C_{\mathrm{uniform}}};\qquad BV(\%)=100\frac{SD(\bar C_{B,k})}{Mean(\bar C_{B,k})}")
        st.markdown("**Pass / action limits:** NEMA standards principally define measurement methods and reporting metrics; they do not establish one universal pass threshold for every scanner and routine QC check. Establish local limits from commissioning data, vendor specifications, applicable regulation/accreditation and a physicist-approved QC programme. If a criterion is not specified for a metric, report the value and trend rather than inventing a pass/fail label. Accreditation criteria must only be applied with the prescribed phantom setup and acquisition/reconstruction protocol.")
    st.markdown("### QC interpretation and system assessment")
    st.markdown("**Do not use one universal pass limit for all PET scanners.** NEMA NU 4 and NU 2 are performance-characterisation methods and define what/how to measure; the acceptance/action limits for routine constancy should be based on commissioning baselines, vendor requirements, local regulations and a physicist-approved QC programme. Apply ACR numerical criteria only to the prescribed ACR phantom and protocol.")
    with st.expander("QC assessment checklist for a real scanner"):
        st.markdown("1. **Daily / routine constancy:** detector status, energy/timing checks, blank/uniformity or vendor-defined system check; inspect sinograms and artefacts.\n2. **Quantitative calibration:** cross-calibrate PET against a traceably measured dose calibrator using a uniform phantom; document activity, reference time, volume, decay correction and residual activity.\n3. **Periodic image quality:** use the correct preclinical NU 4 or clinical phantom and protocol; preserve raw/reconstructed data and exact ROI/VOI definitions.\n4. **Trend and action:** compare with acceptance baseline and control limits; investigate shifts, document corrective action and repeat testing before quantitative studies where required.")
    with st.expander("Connect this QC lesson to the reconstruction generated in Module 4"):
        if "recon_result" not in st.session_state:
            st.info("Run a reconstruction in Module 4 first. Its image and ground-truth phantom will appear here for a generic ROI-based comparison.")
        else:
            rec_truth, rec_sino, rec_img, *_ = st.session_state["recon_result"]
            rec_truth=np.asarray(rec_truth,dtype=float)
            rec_img=np.asarray(rec_img,dtype=float)
            bg_mask=(rec_truth>0.85)&(rec_truth<1.15)
            hot_mask=rec_truth>=1.5
            valid=rec_truth>0
            if np.any(bg_mask):
                scale=float(np.mean(rec_truth[bg_mask]))/max(float(np.mean(rec_img[bg_mask])),1e-12)
                scaled=rec_img*scale
                bg=scaled[bg_mask]
                bg_mean=float(np.mean(bg)); bg_cv=100*float(np.std(bg,ddof=1))/max(abs(bg_mean),1e-12) if len(bg)>1 else 0.0
                bg_bias=100*(bg_mean-float(np.mean(rec_truth[bg_mask])))/max(abs(float(np.mean(rec_truth[bg_mask]))),1e-12)
                rmse=float(np.sqrt(np.mean((scaled[valid]-rec_truth[valid])**2))/max(float(np.mean(rec_truth[valid])),1e-12))
                hot_recovery=float(np.mean(scaled[hot_mask])/max(float(np.mean(rec_truth[hot_mask])),1e-12)) if np.any(hot_mask) else np.nan
                fig,ax=plt.subplots(figsize=(5.2,4.7));ax.imshow(scaled,origin="lower",cmap="inferno")
                ax.contour(bg_mask,levels=[0.5],colors=["cyan"],linewidths=0.8)
                if np.any(hot_mask): ax.contour(hot_mask,levels=[0.5],colors=["lime"],linewidths=0.8)
                ax.set_title("Module 4 reconstruction with background (cyan) and hot (green) ROIs")
                ax.set_xticks([]);ax.set_yticks([]);fig.tight_layout();st.pyplot(fig);plt.close(fig)
                a,b,c,d=st.columns(4)
                a.metric("Background CV",f"{bg_cv:.2f}%")
                b.metric("Background bias",f"{bg_bias:+.2f}%")
                c.metric("Hot-region recovery",f"{hot_recovery:.3f}" if np.isfinite(hot_recovery) else "N/A")
                d.metric("Normalised RMSE",f"{rmse:.3f}")
                st.caption("This ROI comparison reuses the actual generated reconstruction and its digital ground truth. It is a generic educational image-analysis exercise only: it is not a NEMA NU 4, NEMA NU 2 or ACR phantom score and must not be labelled a scanner QC pass.")

    st.markdown("### Key references")
    st.markdown("- [NEMA NU 4-2008 small-animal PET image-quality phantom geometry and analysis](https://jnm.snmjournals.org/content/50/10/1730.full).\n- [ACR PET phantom test and ROI instructions (current revision 5 November 2026)](https://accreditationsupport.acr.org/support/solutions/articles/11000062800-phantom-testing-pet-revised-11-5-2026).\n- [ACR PET phantom scoring criteria (revised 9 October 2025)](https://accreditationsupport.acr.org/support/solutions/articles/11000080933-phantom-criteria-revised-10-9-2025).\n- [EANM routine QC recommendations for PET/CT and PET/MR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068535/).")

# -----------------------------------------------------------------------------
# Knowledge check
# -----------------------------------------------------------------------------
with modules[6]:
    st.subheader("Knowledge check")
    questions = [
        ("Why does a higher-energy positron emitter generally have a larger positron-range contribution to image blurring?", ["The positron can travel further before annihilation", "The annihilation photons have higher energy", "The detector crystals become larger"], 0,
         "Higher-energy positrons generally travel further in matter before annihilation. The annihilation photons remain approximately 511 keV."),
        ("Which coincidence type is produced by photons from two unrelated annihilations paired within the timing window?", ["True", "Scattered", "Random"], 2,
         "Random coincidences arise when unrelated photons are paired within the coincidence timing window."),
        ("What does TOF information improve?", ["Localisation along the LOR", "The positron endpoint energy", "The physical half-life"], 0,
         "TOF narrows the probable annihilation location along the LOR."),
        ("What is a sinogram?", ["A map of projection measurements versus angle and detector position", "A map of radionuclide half-lives", "A detector temperature plot"], 0,
         "A sinogram encodes projection data as a function of projection angle and detector position."),
        ("Why can too many MLEM iterations be undesirable?", ["They change the radionuclide", "Image noise can increase", "They eliminate all partial-volume effects"], 1,
         "As iterations progress, contrast recovery may improve while noise also increases. The trade-off depends on the task and data."),
        ("Which QC test can reveal a global activity-concentration scaling error even when the image looks uniform?", ["Visual uniformity only", "Cross-calibration / quantitative accuracy", "Changing the colour map"], 1,
         "A cross-calibration check compares reconstructed activity concentration with a reference measurement."),
        ("What commonly shortens the ortho-positronium lifetime in tissue?", ["Pick-off annihilation and spin conversion", "An increase in the positron endpoint energy after emission", "The PET detector timing window"], 0,
         "In condensed matter, o-Ps interacts with surrounding molecules. Pick-off annihilation and spin conversion shorten its lifetime; two-photon processes are important in tissue."),
        ("What is the usual spin-statistical ratio of p-Ps to o-Ps among unpolarised positronium formation?", ["1:1", "1:3", "3:1"], 1,
         "The singlet p-Ps state has one spin configuration and the triplet o-Ps state has three, giving a 1:3 ratio under the usual unpolarised spin-statistical assumption."),
    ]
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    for i, (question, choices, correct, explanation) in enumerate(questions):
        st.markdown(f"**{i+1}. {question}**")
        val = st.radio("Choose one answer", choices, key=f"quiz_{i}", label_visibility="collapsed")
        st.session_state.quiz_answers[i] = choices.index(val)
        st.divider()
    if st.button("Mark my answers", type="primary", use_container_width=True):
        score = sum(st.session_state.quiz_answers.get(i, -1) == q[2] for i, q in enumerate(questions))
        st.session_state.quiz_score = score
        st.session_state.quiz_marked = True
    if st.session_state.get("quiz_marked", False):
        score = st.session_state.quiz_score
        st.success(f"You scored {score} out of {len(questions)} ({100*score/len(questions):.0f}%).")
        for i, (question, choices, correct, explanation) in enumerate(questions):
            chosen = st.session_state.quiz_answers.get(i, -1)
            st.markdown(f"**Q{i+1}: {'Correct' if chosen == correct else 'Review'}** — {explanation}")
        if st.button("Reset knowledge check"):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_marked = False
            for i in range(len(questions)):
                st.session_state.pop(f"quiz_{i}", None)
            st.rerun()

st.divider()
st.caption("PET Virtual Laboratory v14 · Teaching demonstrator · simplified 2-D/3-D simulations · Verify all real-world PET measurements and QC against validated protocols. Acronyms: PSF, point-spread function; TOF, time-of-flight; PVE, partial-volume effect; FBP, filtered backprojection; MLEM, maximum-likelihood expectation maximisation; OSEM, ordered-subsets expectation maximisation; NECR, noise-equivalent count rate.")
