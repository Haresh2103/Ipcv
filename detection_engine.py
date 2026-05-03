"""
Road Defect Detection Engine
Algorithms: Laplacian Edge Enhancement + Morphological Skeleton Analysis
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import dataclasses
from typing import Optional


@dataclasses.dataclass
class DetectionResult:
    """Holds all outputs from a single detection run."""
    original: np.ndarray
    laplacian_map: np.ndarray
    skeleton_map: np.ndarray
    defect_mask: np.ndarray
    annotated: np.ndarray
    severity_map: np.ndarray

    # Metrics
    pothole_count: int
    crack_count: int
    total_defect_area_pct: float
    severity_score: float          # 0-100
    severity_label: str            # Low / Medium / High / Critical
    pothole_regions: list
    crack_regions: list
    processing_steps: dict         # intermediate images for step-by-step view


def _classify_severity(score: float) -> str:
    if score < 20:
        return "Low"
    elif score < 45:
        return "Medium"
    elif score < 70:
        return "High"
    else:
        return "Critical"


def _colormap_overlay(gray: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a single-channel float [0,1] map to a colored BGR image."""
    norm = np.clip((gray * 255).astype(np.uint8), 0, 255)
    return cv2.applyColorMap(norm, colormap)


def analyze_image(
    image_bgr: np.ndarray,
    laplacian_ksize: int = 5,
    laplacian_threshold: float = 0.12,
    morph_kernel_size: int = 3,
    min_pothole_area: int = 300,
    min_crack_area: int = 60,
    blur_sigma: float = 1.5,
) -> DetectionResult:
    """
    Full pipeline:
      1. Pre-process (Gaussian blur + CLAHE)
      2. Laplacian edge/texture map  →  detects potholes (blob-like dark regions)
      3. Morphological skeleton       →  detects cracks (elongated thin structures)
      4. Combine masks, label regions, annotate
    """
    h, w = image_bgr.shape[:2]
    steps = {}

    # ── 1. PRE-PROCESSING ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast normalisation across varied lighting
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Gaussian blur to suppress sensor noise before Laplacian
    blurred = cv2.GaussianBlur(gray_eq, (0, 0), blur_sigma)
    steps["1_preprocessed"] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # ── 2. LAPLACIAN EDGE MAP ─────────────────────────────────────────────────
    # Laplacian measures 2nd derivative → highlights abrupt depth/texture changes
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_ksize)
    lap_abs = np.abs(lap)
    lap_norm = lap_abs / (lap_abs.max() + 1e-8)

    # Threshold: high Laplacian magnitude = surface discontinuity → pothole candidate
    lap_thresh = (lap_norm > laplacian_threshold).astype(np.uint8)

    # Morphological close to merge nearby edge fragments into blobs
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size * 3, morph_kernel_size * 3)
    )
    pothole_mask_raw = cv2.morphologyEx(lap_thresh, cv2.MORPH_CLOSE, kernel_close)

    steps["2_laplacian"] = _colormap_overlay(lap_norm, cv2.COLORMAP_HOT)

    # ── 3. MORPHOLOGICAL SKELETON (CRACK DETECTION) ───────────────────────────
    # Adaptive threshold to find dark linear features
    adapt = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=8
    )

    # Remove large blobs (potholes) to keep only thin structures
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    adapt_opened = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Skeletonize: reduce crack regions to single-pixel-wide centrelines
    skel_input = (adapt_opened > 0)
    skeleton = skeletonize(skel_input).astype(np.uint8) * 255

    # Dilate skeleton slightly for visibility & connected-component analysis
    skel_dilated = cv2.dilate(skeleton, kernel_open, iterations=2)
    crack_mask_raw = skel_dilated

    steps["3_skeleton"] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # ── 4. REGION FILTERING ───────────────────────────────────────────────────
    def filter_regions(mask, min_area, max_aspect=None, keep_elongated=False):
        """Label connected components and filter by area/shape."""
        labeled = label(mask > 0)
        good_mask = np.zeros_like(mask)
        regions = []
        for prop in regionprops(labeled):
            if prop.area < min_area:
                continue
            # Aspect ratio filter
            minor = prop.axis_minor_length + 1e-8
            major = prop.axis_major_length + 1e-8
            aspect = major / minor
            if keep_elongated and aspect < 2.5:
                continue
            if max_aspect and aspect > max_aspect:
                continue
            good_mask[labeled == prop.label] = 255
            regions.append(prop)
        return good_mask, regions

    pothole_mask, pothole_regions = filter_regions(
        pothole_mask_raw, min_pothole_area, max_aspect=6.0
    )
    crack_mask, crack_regions = filter_regions(
        crack_mask_raw, min_crack_area, keep_elongated=True
    )

    # Subtract potholes from crack mask (avoid double-counting blobs)
    crack_mask = np.clip(crack_mask.astype(int) - pothole_mask.astype(int), 0, 255).astype(np.uint8)

    steps["4_pothole_mask"] = cv2.cvtColor(pothole_mask, cv2.COLOR_GRAY2BGR)
    steps["4_crack_mask"]   = cv2.cvtColor(crack_mask, cv2.COLOR_GRAY2BGR)

    # ── 5. SEVERITY MAP ───────────────────────────────────────────────────────
    # Weighted combination: pothole weight > crack weight
    severity_float = (
        pothole_mask.astype(float) / 255 * 0.7
        + crack_mask.astype(float) / 255 * 0.3
    )
    # Smooth for a heatmap feel
    severity_smooth = ndimage.gaussian_filter(severity_float, sigma=8)
    severity_map_colored = _colormap_overlay(severity_smooth / (severity_smooth.max() + 1e-8))

    # ── 6. ANNOTATED OUTPUT ───────────────────────────────────────────────────
    annotated = image_bgr.copy()

    # Draw pothole bounding boxes in RED
    labeled_pot = label(pothole_mask > 0)
    for prop in regionprops(labeled_pot):
        if prop.area < min_pothole_area:
            continue
        r0, c0, r1, c1 = prop.bbox
        cv2.rectangle(annotated, (c0, r0), (c1, r1), (0, 0, 220), 2)
        label_text = f"Pothole ({prop.area}px²)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (c0, r0 - th - 6), (c0 + tw + 4, r0), (0, 0, 220), -1)
        cv2.putText(annotated, label_text, (c0 + 2, r0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw crack overlay in YELLOW
    crack_overlay = np.zeros_like(annotated)
    crack_overlay[crack_mask > 0] = (0, 220, 220)
    annotated = cv2.addWeighted(annotated, 1.0, crack_overlay, 0.55, 0)

    # Draw crack bounding boxes in YELLOW
    labeled_cr = label(crack_mask > 0)
    for prop in regionprops(labeled_cr):
        if prop.area < min_crack_area:
            continue
        r0, c0, r1, c1 = prop.bbox
        cv2.rectangle(annotated, (c0, r0), (c1, r1), (0, 200, 200), 1)

    # Pothole fill overlay (semi-transparent red)
    pot_overlay = np.zeros_like(annotated)
    pot_overlay[pothole_mask > 0] = (0, 0, 180)
    annotated = cv2.addWeighted(annotated, 1.0, pot_overlay, 0.35, 0)

    steps["5_annotated"] = annotated

    # ── 7. METRICS ────────────────────────────────────────────────────────────
    total_pixels = h * w
    defect_union = np.clip(pothole_mask.astype(int) + crack_mask.astype(int), 0, 255).astype(np.uint8)
    defect_pct = (defect_union > 0).sum() / total_pixels * 100

    pothole_sev = (pothole_mask > 0).sum() / total_pixels * 100
    crack_sev   = (crack_mask > 0).sum() / total_pixels * 100
    severity_score = min(100.0, pothole_sev * 4.0 + crack_sev * 1.5)

    # Laplacian map for display
    lap_display = _colormap_overlay(lap_norm, cv2.COLORMAP_INFERNO)

    return DetectionResult(
        original=image_bgr,
        laplacian_map=lap_display,
        skeleton_map=cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR),
        defect_mask=defect_union,
        annotated=annotated,
        severity_map=severity_map_colored,
        pothole_count=len(regionprops(labeled_pot)),
        crack_count=len(regionprops(labeled_cr)),
        total_defect_area_pct=round(defect_pct, 2),
        severity_score=round(severity_score, 1),
        severity_label=_classify_severity(severity_score),
        pothole_regions=pothole_regions,
        crack_regions=crack_regions,
        processing_steps=steps,
    )


def generate_sample_road_image(seed: int = 42) -> np.ndarray:
    """Generates a synthetic road image with potholes and cracks for demo."""
    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Base asphalt texture
    noise = rng.integers(60, 110, (h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (5, 5), 2)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Add potholes (dark ellipses with depth shading)
    potholes = [(160, 200, 55, 40), (400, 310, 70, 50), (90, 380, 40, 35)]
    for cx, cy, rx, ry in potholes:
        for r in range(max(rx, ry), 0, -1):
            alpha = 1 - r / max(rx, ry)
            col = int(20 + 60 * (1 - alpha))
            cv2.ellipse(img, (cx, cy), (int(rx * r / max(rx, ry)), int(ry * r / max(rx, ry))),
                        0, 0, 360, (col, col, col), -1)

    # Add cracks (polylines)
    cracks = [
        [(300, 50), (320, 130), (310, 200), (340, 280)],
        [(50, 100), (150, 120), (220, 90), (310, 110)],
        [(450, 350), (490, 380), (520, 370), (560, 410)],
        [(200, 400), (230, 430), (260, 415), (290, 445)],
    ]
    for pts in cracks:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False, (30, 30, 30), rng.integers(1, 3))

    # Random noise grain for realism
    grain = rng.integers(-15, 15, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return img
