"""
提取所有灯条的 gray ROI 原图和方差图，以及角点标注。

输出结构:
  {out_dir}/
    gray_roi/         -- 灯条灰度 ROI 原图 (.png)
    variance_map/     -- 方差图 (.png)
    labels/           -- 角点标注 (.txt, ROI 坐标归一化)

用法:
  python extract_light_data.py [--out_dir ../test/light_data]
"""

import cv2
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from light_corner_corrector import LightCornerCorrector
from lable_generator import (TraditionalArmorDetector, NumberClassifier,
                             extract_number_rois, _compute_expand_factor)


def _save_labelme_json(save_dir, base_name, image_name, top_px, bot_px, h, w):
    """保存 LabelMe 格式的 JSON 标注文件。"""
    data = {
        "version": "6.1.3",
        "flags": {},
        "shapes": [
            {
                "label": "top",
                "points": [[float(top_px[0]), float(top_px[1])]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            },
            {
                "label": "bottom",
                "points": [[float(bot_px[0]), float(bot_px[1])]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    json_path = os.path.join(save_dir, f"{base_name}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def extract_all(dataset_dir, out_dir, model_path, label_path):
    gray_dir = os.path.join(out_dir, "gray_roi")
    var_dir  = os.path.join(out_dir, "variance_map")
    lbl_dir  = os.path.join(out_dir, "labels")
    os.makedirs(gray_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    image_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.bmp'):
                image_files.append(os.path.join(root, f))

    detector = TraditionalArmorDetector(binary_thresh=80)

    # 激进参数：放宽限制以提升召回，宁可多检不可漏检
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
    classifier = NumberClassifier(model_path, label_path, threshold=0.3)

    total = 0
    skipped = 0
    filtered_negative = 0

    for img_path in tqdm(image_files, desc="Extracting"):
        bayer_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if bayer_raw is None:
            continue

        _, _, gray_img, armors, _ = detector.detect(bayer_raw)
        stem = Path(img_path).stem

        # 分类，过滤 class_id == 6 (negative)
        roi_images, armor_infos = extract_number_rois(gray_img, armors)
        valid_armor_indices = set()
        for roi_img, armor_info in zip(roi_images, armor_infos):
            class_id, confidence = classifier.classify_single(roi_img)
            if class_id == 6 or confidence < classifier.threshold:
                filtered_negative += 1
                continue
            # 通过 armor_info 中的 light 引用找到对应 armor 索引
            # extract_number_rois 内会 swap 左右灯条，需双向匹配
            for ai, (l1, l2) in enumerate(armors):
                if ((armor_info['light_1'] is l1 and armor_info['light_2'] is l2) or
                    (armor_info['light_1'] is l2 and armor_info['light_2'] is l1)):
                    valid_armor_indices.add(ai)
                    break

        for ai, (l1, l2) in enumerate(armors):
            if ai not in valid_armor_indices:
                continue

            exp1 = _compute_expand_factor(l1.average_brightness, l2.average_brightness)
            exp2 = _compute_expand_factor(l2.average_brightness, l1.average_brightness)

            for li, (light, expand) in enumerate([(l1, exp1), (l2, exp2)]):
                result = corrector.correct_corners(light, gray_img, bayer_raw, expand)
                variance_roi, _, _, axis, top_c, bot_c = result
                if variance_roi is None:
                    skipped += 1
                    continue

                x, y, w, h = corrector.extractor.expanded_bbox
                gray_roi = gray_img[y:y + h, x:x + w]

                # 基础命名: 原图名_装甲板索引_灯条索引
                base = f"{stem}_armor{ai}_light{li}"

                # gray ROI 原图 (uint8)
                cv2.imwrite(os.path.join(gray_dir, f"{base}.png"), gray_roi)

                # 方差图 (uint8)
                cv2.imwrite(os.path.join(var_dir, f"{base}.png"),
                            variance_roi.astype(np.uint8))

                # 角点标注: 归一化到 [0,1] ROI 坐标
                tl_x, tl_y = axis.top_left
                roi_h, roi_w = gray_roi.shape

                def _norm(pt):
                    return ((pt[0] - tl_x) / roi_w, (pt[1] - tl_y) / roi_h)

                with open(os.path.join(lbl_dir, f"{base}.txt"), 'w') as f:
                    if top_c is not None and bot_c is not None:
                        tn = _norm(top_c)
                        bn = _norm(bot_c)
                        f.write(f"{tn[0]:.6f} {tn[1]:.6f} {bn[0]:.6f} {bn[1]:.6f}\n")
                    else:
                        f.write("\n")

                # LabelMe JSON: 像素坐标 (在 ROI 内)
                if top_c is not None and bot_c is not None:
                    top_px = (top_c[0] - tl_x, top_c[1] - tl_y)
                    bot_px = (bot_c[0] - tl_x, bot_c[1] - tl_y)
                    png_name = f"{base}.png"
                    _save_labelme_json(gray_dir, base, png_name,
                                       top_px, bot_px, roi_h, roi_w)
                    _save_labelme_json(var_dir, base, png_name,
                                       top_px, bot_px, roi_h, roi_w)

                total += 1

    print(f"Done. {total} samples saved, {skipped} skipped (len>85), "
          f"{filtered_negative} filtered (class_id==6/low_conf).")
    print(f"  gray_roi:       {gray_dir}")
    print(f"  variance_map:   {var_dir}")
    print(f"  labels:         {lbl_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="../dataset/competation")
    p.add_argument("--out_dir", default="../test/light_data")
    p.add_argument("--model", default="../model/cnn.onnx")
    p.add_argument("--label_txt", default="../model/label.txt")
    args = p.parse_args()

    extract_all(
        os.path.abspath(args.dataset_dir),
        os.path.abspath(args.out_dir),
        os.path.abspath(args.model),
        os.path.abspath(args.label_txt),
    )
