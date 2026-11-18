"""
mole_detector.py - find mole-like dark spots on skin in a BGR frame (OpenCV).

Pipeline
  1. Skin segmentation (YCrCb), holes filled so dark moles inside skin aren't
     cut out, then eroded so the skin/background edge isn't mistaken for a spot.
  2. Black-hat morphology on the LAB lightness channel -> small regions that
     are darker than their surroundings.
  3. Contour filtering by size, circularity and solidity (drops hairs, creases).
  4. Per-spot descriptors loosely modelled on the dermatology "ABCD" checklist
     (asymmetry, border, colour, diameter) so you can log and compare spots
     over time.

This is a hobby / tracking tool, NOT a diagnostic. The notes it attaches are
geometric observations, not a risk score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    # Black-hat kernel must be larger than the biggest spot you care about (px).
    blackhat_kernel: int = 41
    # How much darker (LAB L units, 0-255) a spot must be than its surround.
    contrast_thresh: int = 16
    min_area_px: int = 25
    max_area_frac: float = 0.03       # of frame area
    min_circularity: float = 0.30     # 4*pi*A / P^2  (hair ~0.05, circle 1.0)
    min_solidity: float = 0.75        # area / convex hull area
    min_skin_frac: float = 0.04       # skip frames with less skin than this
    skin_erode_px: int = 15
    px_per_mm: float | None = None    # set to report diameters in mm
    # Thresholds for the descriptive notes
    asym_note: float = 0.25
    border_note: float = 1.45
    color_note: float = 14.0
    diameter_note_mm: float = 6.0


@dataclass
class Spot:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area_px: float
    diameter_px: float
    diameter_mm: float | None
    contrast: float           # surround L - spot L
    asymmetry: float          # 0 = perfectly symmetric
    border: float             # 1 = perfect circle, higher = more irregular
    color_variation: float    # spread of LAB values inside the spot
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def skin_mask(bgr: np.ndarray, erode_px: int = 15) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (40, 133, 77), (255, 180, 130))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    # Fill holes: dark moles may fall outside the skin colour range.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    big = [c for c in contours if cv2.contourArea(c) > 0.005 * m.size]
    cv2.drawContours(filled, big, -1, 255, thickness=cv2.FILLED)
    if erode_px > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        filled = cv2.erode(filled, ek)
    return filled


def _asymmetry(mask: np.ndarray) -> float:
    """Flip the spot about its two principal axes; mean non-overlap / area."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    h, w = mask.shape
    side = int(math.hypot(h, w)) + 4
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = mask
    R = cv2.getRotationMatrix2D((cx + ox, cy + oy), angle, 1.0)
    R[0, 2] += side / 2 - (cx + ox)
    R[1, 2] += side / 2 - (cy + oy)
    aligned = cv2.warpAffine(canvas, R, (side, side), flags=cv2.INTER_NEAREST)
    area = max(int(np.count_nonzero(aligned)), 1)
    scores = []
    for flip in (0, 1):
        diff = cv2.bitwise_xor(aligned, cv2.flip(aligned, flip))
        scores.append(np.count_nonzero(diff) / (2 * area))
    return float(np.mean(scores))


def detect(bgr: np.ndarray, cfg: DetectorConfig | None = None):
    """Return (spots, skin_mask). spots is a list[Spot]."""
    cfg = cfg or DetectorConfig()
    H, W = bgr.shape[:2]
    skin = skin_mask(bgr, cfg.skin_erode_px)
    if np.count_nonzero(skin) < cfg.min_skin_frac * skin.size:
        return [], skin

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    k = cfg.blackhat_kernel | 1
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, cand = cv2.threshold(bh, cfg.contrast_thresh, 255, cv2.THRESH_BINARY)
    cand = cv2.bitwise_and(cand, skin)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    spots: list[Spot] = []
    max_area = cfg.max_area_frac * H * W
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area_px or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * math.pi * area / perim ** 2
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area else 0
        if circ < cfg.min_circularity or solidity < cfg.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(c)
        pad = max(w, h) // 2 + 4
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(local, [c - [x0, y0]], -1, 255, cv2.FILLED)
        ring = cv2.dilate(local, np.ones((9, 9), np.uint8)) & ~cv2.dilate(local, np.ones((3, 3), np.uint8))
        lab_roi = lab[y0:y1, x0:x1]
        inside = lab_roi[local > 0].astype(np.float32)
        around = lab_roi[ring > 0].astype(np.float32)
        contrast = float(around[:, 0].mean() - inside[:, 0].mean()) if len(around) else 0.0
        core = cv2.erode(local, np.ones((5, 5), np.uint8))
        core_px = lab_roi[core > 0].astype(np.float32) if np.count_nonzero(core) >= 10 else inside
        color_var = float(core_px.std(axis=0).mean())

        (_, _), r = cv2.minEnclosingCircle(c)
        diameter_px = 2 * r
        diameter_mm = diameter_px / cfg.px_per_mm if cfg.px_per_mm else None
        M = cv2.moments(c)
        border = perim ** 2 / (4 * math.pi * area)
        asym = _asymmetry(local)

        notes = []
        if asym > cfg.asym_note:
            notes.append("asymmetric")
        if border > cfg.border_note:
            notes.append("irregular border")
        if color_var > cfg.color_note:
            notes.append("uneven colour")
        if diameter_mm and diameter_mm > cfg.diameter_note_mm:
            notes.append(f">{cfg.diameter_note_mm:g}mm")

        spots.append(Spot(x, y, w, h, M["m10"] / M["m00"], M["m01"] / M["m00"],
                          area, diameter_px, diameter_mm, contrast, asym,
                          border, color_var, notes))
    spots.sort(key=lambda s: s.area_px, reverse=True)
    return spots, skin


def annotate(bgr: np.ndarray, spots, skin=None, show_skin=False) -> np.ndarray:
    out = bgr.copy()
    if show_skin and skin is not None:
        tint = out.copy()
        tint[skin > 0] = (0.85 * tint[skin > 0] + 0.15 * np.array([255, 200, 0])).astype(np.uint8)
        out = tint
    for i, s in enumerate(spots, 1):
        color = (0, 140, 255) if s.notes else (80, 220, 80)
        r = int(s.diameter_px / 2) + 6
        cv2.circle(out, (int(s.cx), int(s.cy)), r, color, 2)
        size = f"{s.diameter_mm:.1f}mm" if s.diameter_mm else f"{s.diameter_px:.0f}px"
        label = f"#{i} {size}" + (" *" if s.notes else "")
        cv2.putText(out, label, (int(s.cx) + r + 3, int(s.cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (int(s.cx) + r + 3, int(s.cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"spots: {len(spots)}   (not a diagnosis)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        sys.exit("usage: python mole_detector.py image.jpg [out.jpg]")
    img = cv2.imread(sys.argv[1])
    spots, skin = detect(img)
    print(json.dumps([s.to_dict() for s in spots], indent=2))
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], annotate(img, spots, skin, show_skin=True))
