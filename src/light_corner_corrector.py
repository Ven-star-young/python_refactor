"""
Light corner re-extraction using variance map analysis.

Port of C++ LightCornerCorrector from rw-vision-main.
The variance map highlights edges/transitions; corner positions are found
by scanning along the symmetry axis for maximum brightness drop.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants (1:1 from C++)
# ---------------------------------------------------------------------------
SCALE_WIDTH = 0.3
SCALE_HEIGHT = 0.06
MIN_EXPAND_PIXELS = 2
MAX_BRIGHTNESS = 10.0
START = 0.7 / 2       # 0.35  -- search start from centroid (fraction of half-length)
END = 1.3 / 2         # 0.65  -- search end
CANDIDATE_SHRETHOLD = 0.5
PASS_OPTIMIZE_WIDTH = 0
MAX_LIGHT_LENGTH = 85


# ---------------------------------------------------------------------------
# SymmetryAxis
# ---------------------------------------------------------------------------

@dataclass
class SymmetryAxis:
    """Represents the symmetry axis of a light bar.

    Coordinates: centroid and direction are in full-image coordinates
    (before change_top_left_to is called) or ROI-local (after).
    top_left is the offset of the ROI in the full image.
    """
    top_left: Tuple[int, int]
    centroid: Tuple[float, float]
    direction: Tuple[float, float]
    mean_val: float

    def change_top_left_to(self, new_top_left: Tuple[int, int]) -> None:
        """Shift centroid so coordinates become relative to new_top_left."""
        dx = self.top_left[0] - new_top_left[0]
        dy = self.top_left[1] - new_top_left[1]
        self.centroid = (self.centroid[0] + dx, self.centroid[1] + dy)
        self.top_left = new_top_left


# ---------------------------------------------------------------------------
# RotatedRectExtractor
# ---------------------------------------------------------------------------

class RotatedRectExtractor:
    """Expands a RotatedRect, computes affine to upright it, extracts ROI.

    Coordinate chain used internally:
      1. Full image (e.g. 1240x1624)
      2. expanded_bbox coords (cropped from full image)
      3. Warped image coords (after cv2.warpAffine with M)
      4. rotated_roi (final upright crop, the "ROI-local" space)
    """

    def __init__(self):
        self.original_rect = None
        self.expanded_rect = None
        self.expanded_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
        self.rotated_size: Tuple[float, float] = (0.0, 0.0)
        self.corrected_angle: float = 0.0
        self.rotated_center: Tuple[float, float] = (0.0, 0.0)
        self.M: Optional[np.ndarray] = None       # 2x3 affine
        self.M_inv: Optional[np.ndarray] = None   # 2x3 inverse affine
        self.rotated_roi: Optional[np.ndarray] = None

    # --- Public API ---

    def init(self, light,
             scale_w: float, scale_h: float,
             min_expand_pixels_width: float, min_expand_pixels_height: float,
             image_size: Tuple[int, int]) -> None:
        """Compute expand + affine matrices.  No image extraction yet.

        Args:
            light: must have .center=(cx,cy), .width, .length, .angle
            image_size: (width, height) of full image
        """
        # 使用 light.width / light.length（而非 light.size），因为 Light 构造器
        # 已经做了宽高交换和角度调整，与 C++ Light(继承 RotatedRect) 一致
        r_rect = ((light.center[0], light.center[1]),
                  (light.width, light.length),
                  light.angle)
        self.original_rect = r_rect

        self.expanded_rect = self._expand_rotated_rect(
            r_rect, scale_w, scale_h,
            min_expand_pixels_width, min_expand_pixels_height,
            image_size)

        bbox = cv2.boundingRect(cv2.boxPoints(self.expanded_rect).astype(np.float32))
        self.expanded_bbox = self._clip_bbox(bbox, image_size)

        self._compute_affine()

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Crop expanded_bbox from image, warp with M, crop to rotated_roi size.

        Returns the upright ROI (a copy).
        """
        x, y, w, h = self.expanded_bbox
        if w <= 1 or h <= 1:
            self.rotated_roi = None
            return None

        roi = image[y:y + h, x:x + w].copy()
        warped = cv2.warpAffine(roi, self.M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

        # Crop to the target upright rect centred at rotated_center
        extract_x = int(self.rotated_center[0] - self.rotated_size[0] / 2)
        extract_y = int(self.rotated_center[1] - self.rotated_size[1] / 2)
        extract_w = int(self.rotated_size[0]) + 1
        extract_h = int(self.rotated_size[1]) + 1

        extract_x = max(0, extract_x)
        extract_y = max(0, extract_y)
        extract_w = min(extract_w, w - extract_x)
        extract_h = min(extract_h, h - extract_y)

        self.rotated_roi = warped[extract_y:extract_y + extract_h,
                                  extract_x:extract_x + extract_w].copy()
        return self.rotated_roi

    def transform_back_point(self, pt_in_roi: Tuple[float, float]) -> Tuple[float, float]:
        """Convert point from rotated_roi coords back to full-image coords."""
        offset_x = self.rotated_center[0] - self.rotated_size[0] / 2.0
        offset_y = self.rotated_center[1] - self.rotated_size[1] / 2.0
        pt_warped = np.array([pt_in_roi[0] + offset_x,
                               pt_in_roi[1] + offset_y, 1.0], dtype=np.float64)
        pt_bbox = self.M_inv @ pt_warped
        return (float(pt_bbox[0] + self.expanded_bbox[0]),
                float(pt_bbox[1] + self.expanded_bbox[1]))

    def transform_back_direction(self, dir_in_roi: Tuple[float, float]) -> Tuple[float, float]:
        """Convert direction vector (rotation only, no translation)."""
        zero = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d = np.array([dir_in_roi[0], dir_in_roi[1], 1.0], dtype=np.float64)
        t_zero = self.M_inv @ zero
        t_d = self.M_inv @ d
        return (float(t_d[0] - t_zero[0]), float(t_d[1] - t_zero[1]))

    # --- Private helpers ---

    @staticmethod
    def _make_rrect(cx, cy, w, h, angle):
        """Plain Python tuple in minAreaRect format: ((cx,cy),(w,h),angle)."""
        return ((float(cx), float(cy)), (float(w), float(h)), float(angle))

    def _expand_rotated_rect(self, rect, scale_w, scale_h,
                              min_expand_w, min_expand_h, image_size):
        (cx, cy), (w, h), angle = rect
        expand_w = max(float(w) * scale_w, float(min_expand_w))
        expand_h = max(float(h) * scale_h, float(min_expand_h))
        new_w = float(w) + 2 * expand_w
        new_h = float(h) + 2 * expand_h
        expanded = self._make_rrect(cx, cy, new_w, new_h, angle)

        # boundingRect accepts plain tuple in OpenCV Python
        rect_arr = cv2.boxPoints(expanded)
        bbox = cv2.boundingRect(rect_arr.astype(np.float32))
        clamped = self._clip_bbox(bbox, image_size)
        if (clamped[2] != bbox[2] or clamped[3] != bbox[3] or
            clamped[0] != bbox[0] or clamped[1] != bbox[1]):
            scale_x = image_size[0] / bbox[2] if bbox[2] > 0 else 1.0
            scale_y = image_size[1] / bbox[3] if bbox[3] > 0 else 1.0
            safe_scale = min(1.0, scale_x, scale_y)
            new_w *= safe_scale
            new_h *= safe_scale
            expanded = self._make_rrect(cx, cy, new_w, new_h, angle)
        return expanded

    @staticmethod
    def _clip_bbox(bbox_xywh, image_size):
        x, y, w, h = bbox_xywh
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_size[0] - x)
        h = min(h, image_size[1] - y)
        return (x, y, w, h)

    def _compute_affine(self):
        w, h = self.expanded_rect[1]
        angle = self.expanded_rect[2]
        self.corrected_angle = angle
        self.rotated_size = (w, h)

        if self.corrected_angle < -45.0:
            self.corrected_angle += 90.0
            self.rotated_size = (h, w)
        elif self.corrected_angle > 45.0:
            self.corrected_angle -= 90.0
            self.rotated_size = (h, w)

        cx_full, cy_full = self.expanded_rect[0]
        self.rotated_center = (cx_full - self.expanded_bbox[0],
                                cy_full - self.expanded_bbox[1])

        self.M = cv2.getRotationMatrix2D(self.rotated_center,
                                          self.corrected_angle, 1.0)
        self.M_inv = cv2.invertAffineTransform(self.M)


# ---------------------------------------------------------------------------
# BayerOptimizer
# ---------------------------------------------------------------------------

class BayerOptimizer:
    """Optimise Bayer demosaicing ratios and compute variance maps.

    Identifies the Bayer pattern empirically, then finds optimal G and B
    channel multipliers that minimise 2x2-window variance along the
    symmetry axis.  The resulting enhanced image is used as input for
    variance-map computation.
    """

    def __init__(self, g_ratio: float = 0.6, b_ratio: float = 0.2):
        self.g_ratio = g_ratio
        self.b_ratio = b_ratio
        self.pattern_map: dict = {}          # {(row%2, col%2): 'R'|'G'|'B'}
        self.original_image: Optional[np.ndarray] = None
        self.image_float: Optional[np.ndarray] = None
        self.sample_points: List[Tuple[int, int]] = []
        self.sample_step: int = 4

    # ------------------------------------------------------------------
    # Pattern identification
    # ------------------------------------------------------------------

    def identify_bayer_pattern(self, image: np.ndarray) -> None:
        """Determine which 2x2 positions are R, G, B by average brightness."""
        rows, cols = image.shape
        brightness = np.zeros((2, 2), dtype=np.float64)
        pixel_count = np.zeros((2, 2), dtype=np.int32)

        for i in range(0, rows - rows % 2, 2):
            for j in range(0, cols - cols % 2, 2):
                for di in range(2):
                    for dj in range(2):
                        if i + di < rows and j + dj < cols:
                            brightness[di, dj] += float(image[i + di, j + dj])
                            pixel_count[di, dj] += 1

        for di in range(2):
            for dj in range(2):
                if pixel_count[di, dj] > 0:
                    brightness[di, dj] /= pixel_count[di, dj]

        max_pos = np.unravel_index(np.argmax(brightness), brightness.shape)
        max_di, max_dj = max_pos

        self.pattern_map.clear()
        for di in range(2):
            for dj in range(2):
                if di == max_di and dj == max_dj:
                    self.pattern_map[(di, dj)] = 'R'
                elif di == 1 - max_di and dj == 1 - max_dj:
                    self.pattern_map[(di, dj)] = 'B'
                else:
                    self.pattern_map[(di, dj)] = 'G'

    def _get_pixel_color(self, row: int, col: int) -> str:
        return self.pattern_map[(row % 2, col % 2)]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def prepare_for_optimization(self, image: np.ndarray,
                                  line_point: Optional[Tuple[float, float]] = None,
                                  line_direction: Optional[Tuple[float, float]] = None) -> None:
        """Build sampling points: along a line or adaptive grid."""
        self.original_image = image
        self.image_float = image.astype(np.float32)
        self.sample_points.clear()
        rows, cols = image.shape[:2]

        use_line = (line_point is not None and line_direction is not None and
                    line_point[0] >= 0 and line_point[1] >= 0 and
                    (line_direction[0] != 0 or line_direction[1] != 0))

        if use_line:
            self._sample_along_line(line_point, line_direction, rows, cols)
            self.sample_step = 1
        else:
            total_pixels = rows * cols
            if total_pixels > 1000000:
                step = 5
            elif total_pixels > 250000:
                step = 3
            else:
                step = 1
            self.sample_step = step
            for i in range(0, rows - 1, step):
                for j in range(0, cols - 1, step):
                    self.sample_points.append((j, i))

    def _sample_along_line(self, point, direction, rows, cols):
        norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        if norm == 0:
            return
        ux, uy = direction[0] / norm, direction[1] / norm
        step_size = 0.5
        seen = set()

        # Positive direction
        t = 0.0
        while True:
            cx, cy = point[0] + t * ux, point[1] + t * uy
            x, y = int(round(cx)), int(round(cy))
            if x < 0 or y < 0 or x >= cols - 1 or y >= rows - 1:
                break
            key = (x, y)
            if key not in seen:
                self.sample_points.append(key)
                seen.add(key)
            t += step_size

        # Negative direction
        t = -step_size
        while True:
            cx, cy = point[0] + t * ux, point[1] + t * uy
            x, y = int(round(cx)), int(round(cy))
            if x < 0 or y < 0 or x >= cols - 1 or y >= rows - 1:
                break
            key = (x, y)
            if key not in seen:
                self.sample_points.append(key)
                seen.add(key)
            t -= step_size

    # ------------------------------------------------------------------
    # Variance-based ratio optimisation
    # ------------------------------------------------------------------

    def fast_compute_2x2_variance(self, g: float, b: float) -> float:
        """Average 2x2-window variance across all sample points."""
        if not self.sample_points:
            return 0.0

        variance_sum = 0.0
        for x, y in self.sample_points:
            pixels = []
            for di in range(2):
                for dj in range(2):
                    row, col = y + di, x + dj
                    color = self._get_pixel_color(row, col)
                    ratio = 1.0 if color == 'R' else (g if color == 'G' else b)
                    val = self.image_float[row, col] / ratio
                    val = np.clip(val, 0.0, 255.0)
                    pixels.append(val)
            arr = np.array(pixels)
            mean = arr.mean()
            var = (arr ** 2).mean() - mean ** 2
            variance_sum += var
        return variance_sum / len(self.sample_points)

    def optimize_g_fast(self, initial_g: float, fixed_b: float) -> float:
        """Three-stage G ratio optimisation."""
        best_g = initial_g
        best_cost = self.fast_compute_2x2_variance(initial_g, fixed_b)

        # Stage 1: coarse (step 0.2)
        for g in np.arange(0.1, 1.01, 0.2):
            cost = self.fast_compute_2x2_variance(g, fixed_b)
            if cost < best_cost:
                best_cost, best_g = cost, g

        # Stage 2: fine (step 0.05, +/- 0.2)
        start = max(0.1, best_g - 0.2)
        end = min(1.0, best_g + 0.2)
        for g in np.arange(start, end + 1e-9, 0.05):
            cost = self.fast_compute_2x2_variance(g, fixed_b)
            if cost < best_cost:
                best_cost, best_g = cost, g

        # Stage 3: golden-section (tolerance 0.01)
        left = max(0.1, best_g - 0.05)
        right = min(1.0, best_g + 0.05)
        phi = 0.618033988749
        tol = 0.01
        while right - left > tol:
            x1 = right - phi * (right - left)
            x2 = left + phi * (right - left)
            c1 = self.fast_compute_2x2_variance(x1, fixed_b)
            c2 = self.fast_compute_2x2_variance(x2, fixed_b)
            if c1 < c2:
                right = x2
                if c1 < best_cost:
                    best_cost, best_g = c1, x1
            else:
                left = x1
                if c2 < best_cost:
                    best_cost, best_g = c2, x2
        return best_g

    def optimize_b_fast(self, fixed_g: float, initial_b: float) -> float:
        """Three-stage B ratio optimisation."""
        best_b = initial_b
        best_cost = self.fast_compute_2x2_variance(fixed_g, initial_b)

        # Stage 1: coarse (step 0.2)
        for b in np.arange(0.1, 1.01, 0.2):
            cost = self.fast_compute_2x2_variance(fixed_g, b)
            if cost < best_cost:
                best_cost, best_b = cost, b

        # Stage 2: fine (step 0.10, +/- 0.2)
        start = max(0.1, best_b - 0.2)
        end = min(1.0, best_b + 0.2)
        for b in np.arange(start, end + 1e-9, 0.10):
            cost = self.fast_compute_2x2_variance(fixed_g, b)
            if cost < best_cost:
                best_cost, best_b = cost, b

        # Stage 3: golden-section (tolerance 0.01)
        left = max(0.1, best_b - 0.10)
        right = min(1.0, best_b + 0.10)
        phi = 0.618033988749
        tol = 0.01
        while right - left > tol:
            x1 = right - phi * (right - left)
            x2 = left + phi * (right - left)
            c1 = self.fast_compute_2x2_variance(fixed_g, x1)
            c2 = self.fast_compute_2x2_variance(fixed_g, x2)
            if c1 < c2:
                right = x2
                if c1 < best_cost:
                    best_cost, best_b = c1, x1
            else:
                left = x1
                if c2 < best_cost:
                    best_cost, best_b = c2, x2
        return best_b

    # ------------------------------------------------------------------
    # Demosaic & variance map
    # ------------------------------------------------------------------

    def apply_ratios(self, image: np.ndarray, g: float, b: float) -> np.ndarray:
        """Demosaic Bayer by dividing each pixel by its channel ratio.

        Returns float32 image, values clipped to [0, 255].
        """
        output = np.zeros_like(image, dtype=np.float32)
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                color = self._get_pixel_color(i, j)
                ratio = 1.0 if color == 'R' else (g if color == 'G' else b)
                corrected = float(image[i, j]) / ratio
                output[i, j] = np.clip(corrected, 0.0, 255.0)
        return output

    def compute_variance_mat(self, image: np.ndarray) -> np.ndarray:
        """Compute 3x3 local variance at each interior pixel.

        Border (1px halo) is filled with the minimum interior variance.
        Output is float32, normalised to [0, 255].
        """
        rows, cols = image.shape
        variance_map = np.zeros((rows, cols), dtype=np.float32)
        min_var = float('inf')

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                patch = image[i - 1:i + 2, j - 1:j + 2].astype(np.float32)
                mean = patch.mean()
                var = (patch ** 2).mean() - mean ** 2
                if var < min_var:
                    min_var = var
                variance_map[i, j] = var

        # Fill border
        variance_map[:, 0] = min_var
        variance_map[:, -1] = min_var
        variance_map[0, :] = min_var
        variance_map[-1, :] = min_var

        # Normalise to [0, 255]
        vmin, vmax = variance_map.min(), variance_map.max()
        if vmax > vmin + 1e-9:
            variance_map = (variance_map - vmin) / (vmax - vmin) * 255.0
        else:
            variance_map.fill(0.0)
        return variance_map

    def optimize_bayer_to_gray(self, bayer_image: np.ndarray,
                                line_point: Optional[Tuple[float, float]] = None,
                                line_direction: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Full pipeline: identify pattern → optimise ratios → apply.

        Returns float32 enhanced image.
        """
        assert bayer_image.dtype == np.uint8, "BayerOptimizer expects uint8 input"

        self.identify_bayer_pattern(bayer_image)
        self.prepare_for_optimization(bayer_image, line_point, line_direction)

        self.g_ratio = self.optimize_g_fast(self.g_ratio, self.b_ratio)
        self.b_ratio = self.optimize_b_fast(self.g_ratio, self.b_ratio)

        result = self.apply_ratios(bayer_image, self.g_ratio, self.b_ratio)
        return result


# ---------------------------------------------------------------------------
# LightCornerCorrector
# ---------------------------------------------------------------------------

class LightCornerCorrector:
    """Orchestrate the full light-corner re-extraction pipeline."""

    def __init__(self):
        self.extractor = RotatedRectExtractor()
        self.bayer_optimizer = BayerOptimizer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def correct_corners(self, light,
                        gray_img: np.ndarray,
                        bayer_img: np.ndarray,
                        expand_factor: float,
                        debug: bool = False,
                        debug_dir: str = "test"):
        """Run the full re-extraction pipeline for one light bar.

        Args:
            light:          Light object
            gray_img:       Properly debayered grayscale (full image)
            bayer_img:      Raw Bayer (full image, single-channel uint8)
            expand_factor:  gamma^2 from brightness ratio
            debug:          If True, save debug panels to debug_dir
            debug_dir:      Output directory for debug images

        Returns:
            (variance_roi, enhanced_roi, bayer_roi, axis, top_corner, bottom_corner)
        """
        if light.length > MAX_LIGHT_LENGTH:
            return (None, None, None, None, None, None)

        min_expand_h = max(MIN_EXPAND_PIXELS * expand_factor, float(MIN_EXPAND_PIXELS))
        min_expand_w = max(MIN_EXPAND_PIXELS * expand_factor, float(MIN_EXPAND_PIXELS))

        if light.width <= PASS_OPTIMIZE_WIDTH:
            return (None, None, None, None, None, None)

        image_size = (gray_img.shape[1], gray_img.shape[0])

        # 1. Init rotated-rect extractor
        self.extractor.init(light, SCALE_WIDTH, SCALE_HEIGHT,
                            min_expand_w, min_expand_h, image_size)

        # 2. Symmetry axis from gray image
        axis = self._find_symmetry_axis_weighted_least_square(gray_img)
        if axis is None or axis.mean_val == 0:
            return (None, None, None, None, None, None)

        # 3. Refine axis direction (always point upward)
        self._refind_axis_direction(axis)

        # 4. Shift centroid to ROI-local coords
        axis.change_top_left_to((self.extractor.expanded_bbox[0],
                                  self.extractor.expanded_bbox[1]))

        # 5. Extract Bayer ROI and apply gamma correction
        x, y, w, h = self.extractor.expanded_bbox
        bayer_roi = bayer_img[y:y + h, x:x + w].copy()
        gamma = np.sqrt(expand_factor)
        if abs(gamma - 1.0) > 1e-1:
            normalized = bayer_roi.astype(np.float32) / 255.0
            corrected = np.power(normalized, gamma)
            bayer_roi = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

        # 6. Optimise Bayer → gray
        enhanced_roi = self.bayer_optimizer.optimize_bayer_to_gray(
            bayer_roi, axis.centroid, axis.direction)

        # 7. Variance map
        variance_roi = self.bayer_optimizer.compute_variance_mat(enhanced_roi)

        # 8. Find corners
        candidates_top: List[Tuple[float, float]] = []
        candidates_bottom: List[Tuple[float, float]] = []
        top_corner = self._find_corner(variance_roi, light, axis, "top", candidates_top)
        bottom_corner = self._find_corner(variance_roi, light, axis, "bottom", candidates_bottom)

        # 9. Update light geometry
        light.top = top_corner
        light.bottom = bottom_corner
        light.center = (axis.centroid[0] + axis.top_left[0],
                         axis.centroid[1] + axis.top_left[1])
        if top_corner is not None and bottom_corner is not None:
            light.length = float(np.linalg.norm(
                np.array(top_corner) - np.array(bottom_corner)))

        # 10. Debug: save visualization panels + full-image context
        if debug:
            self._debug_save(gray_img, bayer_img, x, y, w, h,
                             bayer_roi, enhanced_roi, variance_roi, axis,
                             top_corner, bottom_corner,
                             candidates_top, candidates_bottom,
                             debug_dir)

        return (variance_roi, enhanced_roi, bayer_roi, axis, top_corner, bottom_corner)

    # ------------------------------------------------------------------
    # Debug: 6-panel viz saved to disk
    # ------------------------------------------------------------------

    def _debug_save(self, gray_img, bayer_img, bx, by, bw, bh,
                    bayer_roi, enhanced_roi, variance_roi, axis,
                    top_corner, bottom_corner,
                    candidates_top, candidates_bottom,
                    debug_dir):

        import os, time

        def _to_u8(mat):
            if mat.dtype in (np.float32, np.float64):
                return np.clip(mat, 0, 255).astype(np.uint8)
            return mat

        def _to_bgr(mat):
            u8 = _to_u8(mat)
            if len(u8.shape) == 2:
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            return u8

        def _clip(pt, size_hw):
            return (max(0, min(size_hw[1] - 1, int(pt[0]))),
                    max(0, min(size_hw[0] - 1, int(pt[1]))))

        # ── Row 1: full-image context ──
        # Draw on the debayered color image derived from Bayer
        full_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
        # Overlay the gray image (better visibility)
        full_disp = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Draw expanded_bbox in green
        cv2.rectangle(full_disp, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # Draw top/bottom corners in red, center in green
        tl = axis.top_left
        if top_corner is not None:
            cv2.circle(full_disp, (int(top_corner[0]), int(top_corner[1])),
                       4, (0, 0, 255), -1)
        if bottom_corner is not None:
            cv2.circle(full_disp, (int(bottom_corner[0]), int(bottom_corner[1])),
                       4, (0, 0, 255), -1)
        center = (axis.centroid[0] + tl[0], axis.centroid[1] + tl[1])
        cv2.circle(full_disp, (int(center[0]), int(center[1])),
                   4, (0, 255, 0), -1)
        if top_corner is not None and bottom_corner is not None:
            cv2.line(full_disp,
                     (int(top_corner[0]), int(top_corner[1])),
                     (int(bottom_corner[0]), int(bottom_corner[1])),
                     (0, 255, 0), 2)

        # Resize full image to a reasonable width for viewing
        full_h, full_w = full_disp.shape[:2]
        target_w = 800
        full_scale = target_w / full_w
        full_disp = cv2.resize(full_disp, (target_w, int(full_h * full_scale)))

        # Draw legend text on full image
        cv2.putText(full_disp, "green rect = expanded_bbox", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(full_disp, "red dots = top/bottom corners", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(full_disp, "green dot = center", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # ── Row 2: ROI panels (1:1 from C++) ──
        gray_roi = gray_img[by:by + bh, bx:bx + bw]
        gray_color = _to_bgr(gray_roi)
        bayer_color = _to_bgr(bayer_roi)
        enhanced_color = _to_bgr(enhanced_roi)
        variance_color = _to_bgr(variance_roi)
        variance_original = variance_color.copy()

        scale = 1.0
        if variance_roi.shape[0] < 150:
            scale = 150.0 / variance_roi.shape[0]
            h_new = int(gray_color.shape[0] * scale)
            w_new = int(gray_color.shape[1] * scale)
            gray_color = cv2.resize(gray_color, (w_new, h_new))
            bayer_color = cv2.resize(bayer_color, (w_new, h_new))
            enhanced_color = cv2.resize(enhanced_color, (w_new, h_new))
            variance_color = cv2.resize(variance_color, (w_new, h_new))
            variance_original = cv2.resize(variance_original, (w_new, h_new))

        roi_h, roi_w = variance_color.shape[:2]

        # Draw candidates (green), final corners (red), center (green)
        for cand in candidates_top:
            pt = _clip((cand[0] * scale, cand[1] * scale), (roi_h, roi_w))
            cv2.circle(variance_color, pt, 2, (0, 255, 0), -1)
        for cand in candidates_bottom:
            pt = _clip((cand[0] * scale, cand[1] * scale), (roi_h, roi_w))
            cv2.circle(variance_color, pt, 2, (0, 255, 0), -1)

        if top_corner is not None:
            t_roi = (top_corner[0] - tl[0], top_corner[1] - tl[1])
            cv2.circle(variance_color,
                       _clip((t_roi[0] * scale, t_roi[1] * scale), (roi_h, roi_w)),
                       3, (0, 0, 255), -1)
        if bottom_corner is not None:
            b_roi = (bottom_corner[0] - tl[0], bottom_corner[1] - tl[1])
            cv2.circle(variance_color,
                       _clip((b_roi[0] * scale, b_roi[1] * scale), (roi_h, roi_w)),
                       3, (0, 0, 255), -1)
        c_roi = (center[0] - tl[0], center[1] - tl[1])
        cv2.circle(variance_color,
                   _clip((c_roi[0] * scale, c_roi[1] * scale), (roi_h, roi_w)),
                   3, (0, 255, 0), -1)

        # Add labels above each ROI panel
        def _label(img, text):
            h, w = img.shape[:2]
            # Pad top for label
            label_bar = np.zeros((20, w, 3), dtype=np.uint8)
            cv2.putText(label_bar, text, (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return cv2.vconcat([label_bar, img])

        gray_color   = _label(gray_color,   "gray ROI")
        bayer_color  = _label(bayer_color,  "bayer ROI")
        enhanced_color = _label(enhanced_color, "enhanced")
        variance_original = _label(variance_original, "variance")
        variance_color = _label(variance_color, "variance + candidates + result")

        roi_panels = cv2.hconcat(
            [gray_color, bayer_color, enhanced_color, variance_original, variance_color])

        # ── Vertically stack: full context + ROI panels ──
        # Match widths
        fh, fw = full_disp.shape[:2]
        rh, rw = roi_panels.shape[:2]
        if fw > rw:
            roi_panels = cv2.copyMakeBorder(roi_panels, 0, 0, 0, fw - rw,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif rw > fw:
            full_disp = cv2.copyMakeBorder(full_disp, 0, 0, 0, rw - fw,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Separator line
        sep = np.ones((3, max(fw, rw), 3), dtype=np.uint8) * 100

        combined = cv2.vconcat([full_disp, sep, roi_panels])

        # Save
        os.makedirs(debug_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        out_path = os.path.join(debug_dir, f"corrector_debug_{ts}.png")
        cv2.imwrite(out_path, combined)
        print(f"[debug] saved to {out_path}")

    # ------------------------------------------------------------------
    # Private: axis fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _refind_axis_direction(axis: SymmetryAxis) -> None:
        """Flip direction so it always points upward (y <= 0)."""
        if axis.direction[1] > 0:
            axis.direction = (-axis.direction[0], -axis.direction[1])

    def _find_symmetry_axis_weighted_least_square(self, gray_img: np.ndarray) -> Optional[SymmetryAxis]:
        """Weighted PCA on the upright ROI to find the symmetry axis.

        Pixels are weighted by their intensity. The axis direction is the
        minor eigenvector (least-variance direction, i.e. along the light bar).
        """
        roi = self.extractor.extract_from_image(gray_img)
        if roi is None or roi.size == 0:
            return None

        mean_val = float(roi.mean())

        # Normalise to [0, MAX_BRIGHTNESS]
        roi_f32 = roi.astype(np.float32)
        roi_f32 = cv2.normalize(roi_f32, None, 0, MAX_BRIGHTNESS, cv2.NORM_MINMAX)

        # Build weighted point cloud
        rows, cols = roi_f32.shape
        xs, ys, weights = [], [], []
        for i in range(rows):
            for j in range(cols):
                w = int(roi_f32[i, j])
                if w > 0:
                    xs.append(j)
                    ys.append(i)
                    weights.append(w)

        if len(xs) <= 10:
            return SymmetryAxis((0, 0), (0.0, 0.0), (0.0, 0.0), 0.0)

        xs_arr = np.array(xs, dtype=np.float64)
        ys_arr = np.array(ys, dtype=np.float64)
        w_arr = np.array(weights, dtype=np.float64)

        # Normalise weights
        w_sum = w_arr.sum()
        if w_sum == 0:
            return SymmetryAxis((0, 0), (0.0, 0.0), (0.0, 0.0), 0.0)
        w_arr /= w_sum

        # Weighted centroid
        wx = float(np.sum(xs_arr * w_arr))
        wy = float(np.sum(ys_arr * w_arr))

        # Weighted covariance
        cx = xs_arr - wx
        cy = ys_arr - wy
        a00 = float(np.sum(w_arr * cx * cx))
        a01 = float(np.sum(w_arr * cx * cy))
        a11 = float(np.sum(w_arr * cy * cy))
        A = np.array([[a00, a01], [a01, a11]], dtype=np.float64)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        min_idx = 0 if abs(eigenvalues[0]) < abs(eigenvalues[1]) else 1
        axis_vec = eigenvectors[:, min_idx]
        axis_vec /= np.linalg.norm(axis_vec)

        # Direction in ROI space = perpendicular to eigenvector: (vy, -vx)
        dir_roi = (float(axis_vec[1]), float(-axis_vec[0]))

        # Transform back to full-image coords
        centroid_full = self.extractor.transform_back_point((wx, wy))
        dir_full = self.extractor.transform_back_direction(dir_roi)

        return SymmetryAxis((0, 0), centroid_full, dir_full, mean_val)

    # ------------------------------------------------------------------
    # Private: corner finding
    # ------------------------------------------------------------------

    def _find_corner(self, variance_map: np.ndarray,
                      light,
                      axis: SymmetryAxis,
                      order: str,
                      candidates: List) -> Optional[Tuple[float, float]]:
        """Scan along the symmetry axis for the maximum brightness drop.

        Runs multiple parallel scan lines across the light bar width,
        collects candidates, projects them onto the axis, and averages.
        """
        rows, cols = variance_map.shape

        # --- inline helpers ---
        def _in_image(pt):
            return 0 <= pt[0] < cols and 0 <= pt[1] < rows

        def _bilinear(pt):
            x0 = int(np.floor(pt[0]))
            y0 = int(np.floor(pt[1]))
            x1 = x0 + 1
            y1 = y0 + 1
            if x0 < 0 or x1 >= cols or y0 < 0 or y1 >= rows:
                cx = int(np.clip(round(pt[0]), 0, cols - 1))
                cy = int(np.clip(round(pt[1]), 0, rows - 1))
                return float(variance_map[cy, cx])
            dx = pt[0] - x0
            dy = pt[1] - y0
            eps = 1e-6
            w00 = 1.0 / (dx * dx + dy * dy + eps)
            w01 = 1.0 / (dx * dx + (1 - dy) * (1 - dy) + eps)
            w10 = 1.0 / ((1 - dx) * (1 - dx) + dy * dy + eps)
            w11 = 1.0 / ((1 - dx) * (1 - dx) + (1 - dy) * (1 - dy) + eps)
            sw = w00 + w01 + w10 + w11
            i00 = float(variance_map[y0, x0])
            i01 = float(variance_map[y1, x0])
            i10 = float(variance_map[y0, x1])
            i11 = float(variance_map[y1, x1])
            return (i00 * w00 + i01 * w01 + i10 * w10 + i11 * w11) / sw

        def _project(pt):
            rel_x = pt[0] - axis.centroid[0]
            rel_y = pt[1] - axis.centroid[1]
            proj_len = rel_x * axis.direction[0] + rel_y * axis.direction[1]
            return (axis.centroid[0] + proj_len * axis.direction[0],
                    axis.centroid[1] + proj_len * axis.direction[1])

        # --- main ---
        oper = 1 if order == "top" else -1
        L = light.length
        dx = axis.direction[0] * oper
        dy = axis.direction[1] * oper

        half_n = int(round((light.width - 2) / 2.0))
        half_n = max(2, min(half_n, 5))

        candidates.clear()

        for i in range(-half_n, half_n + 1):
            x0 = axis.centroid[0] + L * START * dx + i * 0.5
            y0 = axis.centroid[1] + L * START * dy

            prev = np.array([x0, y0])
            prev_val = _bilinear(prev)
            corner = prev.copy()
            max_diff = 0.0

            step_x = dx / 4.0
            step_y = dy / 4.0
            max_dist = L * (END - START)
            step_norm = np.sqrt(step_x ** 2 + step_y ** 2)
            if step_norm == 0:
                continue
            max_steps = int(max_dist / step_norm)

            for s in range(1, max_steps + 1):
                cur = np.array([x0 + s * step_x, y0 + s * step_y])
                if not _in_image((int(cur[0]), int(cur[1]))):
                    break
                cur_val = _bilinear(cur)
                diff = cur_val - prev_val
                if diff > max_diff:
                    max_diff = diff
                    corner = prev.copy()
                prev = cur
                prev_val = cur_val

            candidates.append((float(corner[0]), float(corner[1])))

        if not candidates:
            return None

        # Sort by distance from centroid, descending
        cx, cy = axis.centroid
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2, reverse=True)

        max_dist = float(np.sqrt((candidates[0][0] - cx) ** 2 + (candidates[0][1] - cy) ** 2))
        result = np.array([0.0, 0.0])
        valid_count = 0
        for cand in candidates:
            dist = float(np.sqrt((cand[0] - cx) ** 2 + (cand[1] - cy) ** 2))
            if dist < max_dist - CANDIDATE_SHRETHOLD:
                break
            proj = _project(cand)
            result += np.array(proj)
            valid_count += 1

        if valid_count == 0:
            return None

        result /= valid_count
        # Convert from ROI-local to full-image coords
        result += np.array([axis.top_left[0], axis.top_left[1]])
        return (float(result[0]), float(result[1]))
