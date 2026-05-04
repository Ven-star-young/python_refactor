"""
检查生成的灯条 ROI 和方差图。

显示布局:
  ┌─────────────────────────────────┐
  │  原图 (含 expanded_bbox 标注)    │
  ├────────────────┬────────────────┤
  │  gray ROI      │  variance map  │
  │  (含角点标注)   │  (含角点标注)   │
  └────────────────┴────────────────┘

操作: n=下一张  p=上一张  q=退出

用法:
  python view_light_data.py --data_dir ../test/light_data
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from light_corner_corrector import LightCornerCorrector
from lable_generator import (TraditionalArmorDetector, NumberClassifier,
                             extract_number_rois, _compute_expand_factor)


def load_labels(lbl_dir):
    """解析 labels 目录，按原图分组。

    文件名格式: {stem}_armor{N}_light{M}.txt
    内容格式: top_x top_y bottom_x bottom_y
    """
    samples = []
    for f in sorted(os.listdir(lbl_dir)):
        if not f.endswith('.txt'):
            continue
        path = os.path.join(lbl_dir, f)
        with open(path) as fh:
            line = fh.read().strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        tn = (float(parts[0]), float(parts[1]))
        bn = (float(parts[2]), float(parts[3]))

        base = f.replace('.txt', '')
        # 反推: {stem}_armor{N}_light{M} → stem, armor_idx, light_idx
        # stem 可能包含下划线，找最后一个 _armor
        idx = base.rfind('_armor')
        if idx < 0:
            continue
        stem = base[:idx]
        rest = base[idx + 1:]  # armor{N}_light{M}
        try:
            armor_str, light_str = rest.split('_')
            armor_idx = int(armor_str.replace('armor', ''))
            light_idx = int(light_str.replace('light', ''))
        except ValueError:
            continue
        samples.append((path, stem, armor_idx, light_idx, tn, bn))
    return samples


def find_image(dataset_dir, stem):
    """在 dataset 目录中搜索原始 BMP。"""
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f == f"{stem}.bmp":
                return os.path.join(root, f)
    return None


def run(light_data_dir, dataset_dir):
    gray_dir = os.path.join(light_data_dir, "gray_roi")
    var_dir = os.path.join(light_data_dir, "variance_map")
    lbl_dir = os.path.join(light_data_dir, "labels")

    samples = load_labels(lbl_dir)
    if not samples:
        print("No samples found.")
        return

    # 按原图分组，缓存检测结果
    detector = TraditionalArmorDetector(binary_thresh=80)
    detector.l_params.min_ratio = 0.005
    detector.l_params.max_ratio = 0.8
    detector.l_params.max_angle = 60
    detector.l_params.min_length = 6
    detector.l_params.min_width = 1
    detector.a_params.min_light_ratio = 0.5
    detector.a_params.min_small_center_distance = 0.5
    detector.a_params.max_small_center_distance = 5.0
    detector.a_params.min_large_center_distance = 2.0
    detector.a_params.max_large_center_distance = 8.0
    detector.a_params.max_angle = 60
    corrector = LightCornerCorrector()

    cache = {}  # stem -> (bayer_raw, gray_img, armors)

    idx = 0
    win_name = "Light Data Viewer — [n]ext [p]rev [q]uit"

    def draw_sample(sample):
        _, stem, armor_idx, light_idx, tn, bn = sample

        # 加载 / 缓存原图检测结果
        if stem not in cache:
            img_path = find_image(dataset_dir, stem)
            if img_path is None:
                print(f"  Image not found: {stem}.bmp")
                return None
            bayer_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if bayer_raw is None:
                return None
            _, _, gray_img, armors, _ = detector.detect(bayer_raw)
            cache[stem] = (bayer_raw, gray_img, armors)
        else:
            bayer_raw, gray_img, armors = cache[stem]

        if armor_idx >= len(armors):
            print(f"  armor_idx {armor_idx} out of range ({len(armors)} armors)")
            return None

        l1, l2 = armors[armor_idx]
        lights = [l1, l2]
        if light_idx >= len(lights):
            print(f"  light_idx {light_idx} out of range")
            return None

        target_light = lights[light_idx]
        other_light = lights[1 - light_idx]

        expand = _compute_expand_factor(target_light.average_brightness,
                                         other_light.average_brightness)

        result = corrector.correct_corners(target_light, gray_img, bayer_raw, expand)
        variance_roi, _, _, axis, top_c, bot_c = result
        if variance_roi is None:
            print(f"  corrector early exit (len={target_light.length:.0f})")
            return None

        bx, by, bw, bh = corrector.extractor.expanded_bbox
        gray_roi = gray_img[by:by + bh, bx:bx + bw]

        # --- 原图 ---
        full_bgr = cv2.cvtColor(bayer_raw, cv2.COLOR_BayerBG2BGR)
        full_disp = full_bgr.copy()
        cv2.rectangle(full_disp, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        tl = axis.top_left
        if top_c is not None:
            cv2.circle(full_disp, (int(top_c[0]), int(top_c[1])), 6, (0, 0, 255), -1)
        if bot_c is not None:
            cv2.circle(full_disp, (int(bot_c[0]), int(bot_c[1])), 6, (0, 0, 255), -1)
        center = (axis.centroid[0] + tl[0], axis.centroid[1] + tl[1])
        cv2.circle(full_disp, (int(center[0]), int(center[1])), 4, (0, 255, 0), -1)
        if top_c is not None and bot_c is not None:
            cv2.line(full_disp,
                     (int(top_c[0]), int(top_c[1])),
                     (int(bot_c[0]), int(bot_c[1])),
                     (0, 255, 0), 2)

        fh, fw = full_disp.shape[:2]
        target_w = 900
        scale = target_w / fw
        full_disp = cv2.resize(full_disp, (target_w, int(fh * scale)))

        cv2.putText(full_disp, f"{stem}  armor={armor_idx}  light={light_idx}  "
                    f"({idx+1}/{len(samples)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(full_disp, "green=bbox  red=corners  green_dot=center",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- gray ROI + variance (放大到固定高度) ---
        ROI_TARGET_H = 400

        def annotate_roi(img, title):
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            h, w = disp.shape[:2]
            # 放大
            scale_r = ROI_TARGET_H / h
            disp = cv2.resize(disp, (int(w * scale_r), ROI_TARGET_H),
                              interpolation=cv2.INTER_NEAREST)
            h2, w2 = disp.shape[:2]
            if top_c is not None:
                cv2.circle(disp, (int(tn[0] * w2), int(tn[1] * h2)), 6, (0, 0, 255), -1)
            if bot_c is not None:
                cv2.circle(disp, (int(bn[0] * w2), int(bn[1] * h2)), 6, (0, 0, 255), -1)
            if top_c is not None and bot_c is not None:
                cv2.line(disp,
                         (int(tn[0] * w2), int(tn[1] * h2)),
                         (int(bn[0] * w2), int(bn[1] * h2)),
                         (0, 255, 0), 2)
            cv2.putText(disp, title, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return disp

        gray_disp = annotate_roi(gray_roi, "gray ROI")
        var_disp = annotate_roi(variance_roi.astype(np.uint8), "variance map")

        # 对齐高度
        gh = gray_disp.shape[0]
        vh = var_disp.shape[0]
        if gh < vh:
            gray_disp = cv2.copyMakeBorder(gray_disp, 0, vh - gh, 0, 0,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif vh < gh:
            var_disp = cv2.copyMakeBorder(var_disp, 0, gh - vh, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        roi_panels = cv2.hconcat([gray_disp, var_disp])

        # 统一宽度，上下拼接
        max_w = max(full_disp.shape[1], roi_panels.shape[1])
        if full_disp.shape[1] < max_w:
            full_disp = cv2.copyMakeBorder(full_disp, 0, 0, 0, max_w - full_disp.shape[1],
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if roi_panels.shape[1] < max_w:
            roi_panels = cv2.copyMakeBorder(roi_panels, 0, 0, 0, max_w - roi_panels.shape[1],
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))

        combined = cv2.vconcat([full_disp, roi_panels])
        return combined

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, 800)

    while 0 <= idx < len(samples):
        print(f"[{idx+1}/{len(samples)}] {samples[idx][1]} armor{samples[idx][2]} light{samples[idx][3]}")
        combined = draw_sample(samples[idx])
        if combined is not None:
            cv2.imshow(win_name, combined)
        else:
            idx += 1
            continue

        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            idx = min(idx + 1, len(samples) - 1)
        elif key == ord('p'):
            idx = max(idx - 1, 0)
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../test/light_data")
    p.add_argument("--dataset_dir", default="../dataset/competation")
    args = p.parse_args()

    run(os.path.abspath(args.data_dir), os.path.abspath(args.dataset_dir))
