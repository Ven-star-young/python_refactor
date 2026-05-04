"""
使用传统算法(灯条检测+匹配)自动标注装甲板数据
生成 YOLOv8-pose 格式的标注文件
"""

#=================================================================================
# 灯条和装甲板检测器
#=================================================================================

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil

class Light:
    """灯条类"""
    def __init__(self, rotated_rect):
        self.center = rotated_rect[0]
        self.size = rotated_rect[1]
        self.angle = rotated_rect[2]
        self.color = None  # 预留颜色属性 蓝色:0 红色:1
        
        # 计算长宽
        if self.size[0] < self.size[1]:
            self.width = self.size[0]
            self.length = self.size[1]
        else:
            self.width = self.size[1]
            self.length = self.size[0]
            self.angle += 90
        
        # 平均亮度（灰度图轮廓内均值）
        self.average_brightness = 0.0

        # 计算顶点 - 使用box顶点取平均(修复:取对边中点)
        box = cv2.boxPoints(rotated_rect)  # 获取旋转矩形的4个顶点
        box = np.array(box)

        # 按Y坐标排序,分为上下两组
        sorted_box = box[np.argsort(box[:, 1])]

        # 上边两个顶点的中点
        self.top = tuple(((sorted_box[0] + sorted_box[1]) / 2).astype(float))

        # 下边两个顶点的中点
        self.bottom = tuple(((sorted_box[2] + sorted_box[3]) / 2).astype(float))

class LightParams:
    """灯条筛选参数(参考C++代码)"""
    min_ratio = 0.01      # 最小宽长比
    max_ratio = 0.6      # 最大宽长比(从0.55降低到0.4,更严格)
    max_angle = 45       # 最大倾斜角度
    min_length = 10      # 最小长度(增加到10)
    min_width = 2        # 最小宽度

class ArmorParams:
    """装甲板筛选参数"""
    min_light_ratio = 0.7
    min_small_center_distance = 0.8
    max_small_center_distance = 3.2
    min_large_center_distance = 3.2
    max_large_center_distance = 5.5
    max_angle = 45
    max_armor_width = 300  # 装甲板最大宽度(像素) - 从200调整到300
    max_armor_height = 150  # 装甲板最大高度(像素) - 从200调整到150

class TraditionalArmorDetector:
    """传统装甲板检测器"""

    def __init__(self, binary_thresh=100):
        self.binary_thresh = binary_thresh
        self.l_params = LightParams()
        self.a_params = ArmorParams()
    
    def preprocess_image(self, bayer_raw):
        """图像预处理 - 二值化。从 Bayer 原图生成 BGR、灰度图和二值图"""
        bgr_img = cv2.cvtColor(bayer_raw, cv2.COLOR_BayerBG2BGR)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, self.binary_thresh, 255, cv2.THRESH_BINARY)
        return binary, bgr_img, gray_img
    
    def find_lights(self, binary_img, bgr_img, gray_img=None) -> List[Light]:
        """查找灯条"""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lights = []
        for contour in contours:
            if len(contour) < 5:       
                continue
            
            r_rect = cv2.minAreaRect(contour)
            light = Light(r_rect)
            
            if not self.is_light(light):
                continue

            # 计算灯条颜色 - 基于ROI区域内的R/B通道统计
            sum_r, sum_g, sum_b = 0, 0, 0
            pixel_count = 0
            
            # 获取轮廓的外接矩形
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            
            # 提取ROI区域
            roi = bgr_img[y:y+h, x:x+w]
            
            # 遍历ROI中的每个像素
            for i in range(h):
                for j in range(w):
                    # 检查点是否在轮廓内（使用原图坐标系）
                    point = (j + x, i + y)
                    if cv2.pointPolygonTest(contour, point, False) >= 0:
                        # 累加RGB值 (OpenCV格式是BGR)
                        b, g, r = roi[i, j]
                        sum_r += int(r)
                        sum_g += int(g)
                        sum_b += int(b)
                        pixel_count += 1
            
            # 判断灯条颜色 (红色 vs 蓝色)
            if pixel_count > 0:
                avg_r = sum_r / pixel_count
                avg_b = sum_b / pixel_count
                # 如果红色分量更大，标记为红色灯条，否则为蓝色
                light.color = 1 if avg_r > avg_b else 0
            else:
                light.color = -1

            # 计算平均亮度（用于 expand_factor）
            if gray_img is not None:
                light.average_brightness = self._compute_light_average_brightness(
                    contour, gray_img)

            lights.append(light)

        return lights

    
    def is_light(self, light: Light) -> bool:
        """判断是否为有效灯条"""
        if light.length == 0:
            return False
        
        # 长宽比判断 - 灯条应该是细长的
        ratio = light.width / light.length
        ratio_ok = self.l_params.min_ratio < ratio < self.l_params.max_ratio
        
        # 角度判断
        angle_ok = abs(light.angle) < self.l_params.max_angle
        
        # 尺寸判断 - 确保灯条足够长且有一定宽度
        size_ok = light.length >= self.l_params.min_length and light.width >= self.l_params.min_width
        
        return ratio_ok and angle_ok and size_ok

    def _compute_light_average_brightness(self, contour, gray_img):
        """灰度轮廓内的均值亮度，用于 expand_factor 计算"""
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        if x < 0 or y < 0 or x + w > gray_img.shape[1] or y + h > gray_img.shape[0]:
            return 0.0
        roi = gray_img[y:y + h, x:x + w]
        shifted = contour - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, shifted.astype(np.int32), 255)
        return cv2.mean(roi, mask=mask)[0]

    def match_lights(self, lights: List[Light]) -> List[Tuple[Light, Light]]:
        """匹配灯条形成装甲板"""
        armors = []
        
        for i, light_1 in enumerate(lights):
            for light_2 in lights[i+1:]:
                if (light_1.color is not None and light_2.color is not None and
                    light_1.color != light_2.color):
                    continue
                if self.is_armor(light_1, light_2):
                    armors.append((light_1, light_2))
        
        return armors
    
    def is_armor(self, light_1: Light, light_2: Light) -> bool:
        """判断两个灯条是否能组成装甲板"""
        # 1. 长度比例 - 两个灯条长度应该相近
        light_length_ratio = min(light_1.length, light_2.length) / max(light_1.length, light_2.length)
        if light_length_ratio < self.a_params.min_light_ratio:
            return False
        
        # 2. 灯条角度差 - 两个灯条应该接近平行
        angle_diff = abs(light_1.angle - light_2.angle)
        # 处理角度跨越180度的情况
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 15:  # 两灯条角度差不超过15度
            return False
        
        # 3. 中心距离检查
        avg_light_length = (light_1.length + light_2.length) / 2
        center_distance = np.linalg.norm(
            np.array(light_1.center) - np.array(light_2.center)
        ) / avg_light_length
        
        center_distance_ok = (
            (self.a_params.min_small_center_distance <= center_distance < self.a_params.max_small_center_distance) or
            (self.a_params.min_large_center_distance <= center_distance < self.a_params.max_large_center_distance)
        )
        
        if not center_distance_ok:
            return False
        
        # 4. 两灯条中心连线角度 - 应该接近水平(修复)
        diff = np.array(light_1.center) - np.array(light_2.center)
        center_angle = abs(np.arctan2(diff[1], diff[0])) / np.pi * 180
        # 只保留接近水平的: 0-45度 或 135-180度
        angle_ok = center_angle < self.a_params.max_angle or center_angle > (180 - self.a_params.max_angle)
        if not angle_ok:
            return False
        
        # 5. ROI大小检查
        point_distance_top = np.linalg.norm(np.array(light_1.top) - np.array(light_2.top))
        point_distance_bottom = np.linalg.norm(np.array(light_1.bottom) - np.array(light_2.bottom))
        
        if point_distance_top <= 16 or point_distance_bottom <= 16:
            return False
        
        # # 6. 装甲板尺寸限制
        # all_points = np.array([light_1.top, light_1.bottom, light_2.top, light_2.bottom])
        # armor_width = np.max(all_points[:, 0]) - np.min(all_points[:, 0])
        # armor_height = np.max(all_points[:, 1]) - np.min(all_points[:, 1])
        
        # # 宽度和高度限制
        # size_ok = (armor_width <= self.a_params.max_armor_width and 
        #         armor_height <= self.a_params.max_armor_height)
        
        # if not size_ok:
        #     return False
        
        # # 7. 长宽比限制(修复:允许接近正方形)
        # if armor_width > 0 and armor_height > 0:
        #     armor_aspect_ratio = armor_width / armor_height
        #     aspect_ratio_ok = 0.8 < armor_aspect_ratio < 5.0  # 从1.0改为0.8,允许接近正方形
        # else:
        #     aspect_ratio_ok = False
        
        return True
    
    def detect(self, img):
        """检测装甲板"""
        binary_img, bgr_img, gray_img = self.preprocess_image(img)
        lights = self.find_lights(binary_img, bgr_img, gray_img)
        armors = self.match_lights(lights)

        return binary_img, bgr_img, gray_img, armors, lights
    
#==================================================================================
# roi提取
#==================================================================================


def extract_number_rois(src_img, armors):
    """
    提取装甲板数字ROI区域
    参考C++代码的extractNumbers方法
    
    Args:
        src_img: 原始灰度图像
        armors: 装甲板列表 [(light_1, light_2), ...]
    
    Returns:
        roi_images: 提取的ROI图像列表
        armor_infos: 装甲板信息列表(用于后续标注)
    """
    # 参数设置(参考C++代码)
    LIGHT_LENGTH = 12          # 变换后灯条长度
    WARP_HEIGHT = 28           # 变换后图像高度
    SMALL_ARMOR_WIDTH = 32     # 小装甲板宽度
    LARGE_ARMOR_WIDTH = 54     # 大装甲板宽度
    ROI_SIZE = (20, 28)        # 数字ROI尺寸 (width, height)
    
    roi_images = []
    armor_infos = []
    
    for light_1, light_2 in armors:
        # 确保light_1在左边
        if light_1.center[0] > light_2.center[0]:
            light_1, light_2 = light_2, light_1
        
        # 判断装甲板类型(根据归一化距离)
        avg_light_length = (light_1.length + light_2.length) / 2
        center_distance = np.linalg.norm(
            np.array(light_1.center) - np.array(light_2.center)
        ) / avg_light_length
        
        # 根据距离判断大小板
        is_large = center_distance >= 3.2
        warp_width = LARGE_ARMOR_WIDTH if is_large else SMALL_ARMOR_WIDTH
        
        # 源图像四个顶点(按照C++代码的顺序: left_bottom, left_top, right_top, right_bottom)
        lights_vertices = np.float32([
            light_1.bottom,  # 左下
            light_1.top,     # 左上
            light_2.top,     # 右上
            light_2.bottom   # 右下
        ])
        
        # 目标图像四个顶点
        top_light_y = (WARP_HEIGHT - LIGHT_LENGTH) / 2 - 1
        bottom_light_y = top_light_y + LIGHT_LENGTH
        
        target_vertices = np.float32([
            [0, bottom_light_y],                    # 左下
            [0, top_light_y],                       # 左上
            [warp_width - 1, top_light_y],          # 右上
            [warp_width - 1, bottom_light_y]        # 右下
        ])
        
        # 透视变换
        rotation_matrix = cv2.getPerspectiveTransform(lights_vertices, target_vertices)
        number_image = cv2.warpPerspective(src_img, rotation_matrix, (warp_width, WARP_HEIGHT))
        
        # 裁剪ROI区域(从中心裁剪)
        roi_x = (warp_width - ROI_SIZE[0]) // 2
        roi_y = 0
        number_roi = number_image[roi_y:roi_y + ROI_SIZE[1], roi_x:roi_x + ROI_SIZE[0]]
        
        # 保存ROI图像和装甲板信息
        roi_images.append(number_roi)
        armor_infos.append({
            'light_1': light_1,
            'light_2': light_2,
            'is_large': is_large,
            'warp_image': number_image,  # 保存完整变换图像用于可视化
            'roi': number_roi
        })
    
    return roi_images, armor_infos


#==================================================================================
# 神经网分类
#==================================================================================


import onnxruntime
import json
from light_corner_corrector import LightCornerCorrector

class NumberClassifier:
    """
    装甲板数字分类器
    参考C++代码的classify方法
    """
    def __init__(self, model_path, label_path, threshold=0.7):
        """
        Args:
            model_path: ONNX模型路径
            label_path: 标签文件路径(每行一个类别名)
            threshold: 置信度阈值
        """
        self.threshold = threshold
        
        # 加载ONNX模型
        self.ort_session = onnxruntime.InferenceSession(model_path)
        
        # 加载类别名
        self.class_names = []
        with open(label_path, 'r') as f:
            for line in f:
                self.class_names.append(line.strip())
        
        print(f"加载模型: {model_path}")
        print(f"类别数量: {len(self.class_names)}")
        print(f"类别: {self.class_names}")
        
        # 打印模型输入信息
        input_info = self.ort_session.get_inputs()[0]
        print(f"\n模型输入名称: {input_info.name}")
        print(f"模型期望形状: {input_info.shape}")
    
    def preprocess(self, roi_img):
        """
        图像预处理 - 完全复刻C++的blobFromImage行为
        
        C++代码:
        cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(28, 20), ...)
        
        注意:
        1. ROI输入是 (28, 20) 的灰度图 - (height, width)
        2. cv::Size(28, 20) 表示 width=28, height=20，会先resize
        3. 最终blob形状: (1, 1, 20, 28) - (batch, channels, height, width)
        """
        # 1. 直方图均衡化
        img_eq = cv2.equalizeHist(roi_img)
        
        # 2. Resize到指定尺寸 (模拟 cv::Size(28, 20) 的效果)
        # cv::Size(width, height) = cv::Size(28, 20)
        # OpenCV resize 参数是 (width, height)
        img_resized = cv2.resize(img_eq, (28, 20), interpolation=cv2.INTER_LINEAR)
        # 现在 img_resized.shape = (20, 28) - (height, width)
        
        # 3. 归一化到[0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 4. 转换为NCHW格式 (batch, channels, height, width)
        # (20, 28) -> (1, 1, 20, 28)
        blob = img_normalized[np.newaxis, np.newaxis, :, :]
        
        return blob
    
    def classify_single(self, roi_img):
        """
        分类单个ROI图像
        
        Returns:
            class_name: 类别名
            confidence: 置信度
        """
        # 预处理
        blob = self.preprocess(roi_img)
        
        # 推理
        ort_inputs = {self.ort_session.get_inputs()[0].name: blob}
        outputs = self.ort_session.run(None, ort_inputs)[0]
        
        # Softmax(参考C++代码)
        max_prob = np.max(outputs)
        softmax_prob = np.exp(outputs - max_prob)
        softmax_prob = softmax_prob / np.sum(softmax_prob)
        
        # 获取最大置信度的类别
        class_id = np.argmax(softmax_prob)
        confidence = softmax_prob[0, class_id]
        return class_id, confidence

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def _compute_expand_factor(this_brightness, other_brightness):
    """计算 expand_factor，用于角点重提取的 ROI 扩张和 gamma 校正。

    C++ 公式:
        gamma = log2(this/255) / log2(other/255)
        gamma = max(gamma, 1.0)
        expand_factor = gamma^2
    """
    eps = 0.1
    b_this = max(this_brightness, eps)
    b_other = max(other_brightness, eps)
    if b_this <= eps or b_other <= eps:
        return 1.0
    gamma = np.log2(b_this / 255.0) / np.log2(b_other / 255.0)
    gamma = max(gamma, 1.0)
    return gamma ** 2


def get_one_image_label(image_bayer, detector, classifier,
                        corrector=None, is_debug=False):
    """对单张图像运行检测 + 分类 + 角点重提取，返回标注数据。

    Args:
        image_bayer:  原始 Bayer 灰度图
        detector:     TraditionalArmorDetector
        classifier:   NumberClassifier
        corrector:    LightCornerCorrector (可选，None 则跳过重提取)

    Returns:
        (classified_data, variance_data)
    """
    height, width = image_bayer.shape

    binary_img, bgr_img, gray_img, armors, lights = detector.detect(image_bayer)

    roi_images, armor_infos = extract_number_rois(gray_img, armors)

    classified_armors = []
    for roi_img, armor_info in zip(roi_images, armor_infos):
        class_id, confidence = classifier.classify_single(roi_img)
        if confidence >= classifier.threshold:
            classified_armors.append((armor_info, class_id, confidence))

    # --- 组装分类输出 ---
    classified_data = []
    for armor_info, class_id, confidence in classified_armors:
        light_1 = armor_info['light_1']
        light_2 = armor_info['light_2']

        # 关键点使用重提取后的角点（若已运行）
        l1_top_norm = (light_1.top[0] / width, light_1.top[1] / height)
        l1_bot_norm = (light_1.bottom[0] / width, light_1.bottom[1] / height)
        l2_top_norm = (light_2.top[0] / width, light_2.top[1] / height)
        l2_bot_norm = (light_2.bottom[0] / width, light_2.bottom[1] / height)

        keypoints = [
            l1_top_norm,
            l1_bot_norm,
            l2_bot_norm,
            l2_top_norm,
        ]

        x_center = (light_1.center[0] + light_2.center[0]) / 2 / width
        y_center = (light_1.center[1] + light_2.center[1]) / 2 / height
        bbox_width = abs(light_2.center[0] - light_1.center[0]) * 1.5 / width
        bbox_height = (light_1.length + light_2.length) / 2 * 2.0 / height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        bbox_width = max(0, min(1, bbox_width))
        bbox_height = max(0, min(1, bbox_height))

        classified_data.append({
            'light_1': light_1,
            'light_2': light_2,
            'class_id': class_id,
            'bbox': (x_center, y_center, bbox_width, bbox_height),
            'keypoints': keypoints,
            'color': light_1.color if light_1.color == light_2.color else -1
        })
    variance_data = []

    # --- 角点重提取 ---
    if corrector is not None:
        for data in classified_data:
            light_1 = data['light_1']
            light_2 = data['light_2']
            class_id = data['class_id']

            if class_id == 6:  # negative 样本不进行重提取
                continue
            expand_1 = _compute_expand_factor(light_1.average_brightness,
                                               light_2.average_brightness)
            expand_2 = _compute_expand_factor(light_2.average_brightness,
                                               light_1.average_brightness)

            for light_idx, (light, expand) in enumerate(
                [(light_1, expand_1), (light_2, expand_2)]):

                result = corrector.correct_corners(
                    light, gray_img, image_bayer, expand, is_debug)
                (variance_roi, enhanced_roi, bayer_roi,
                 axis, top_corner, bottom_corner) = result

                if variance_roi is not None:
                    variance_data.append({
                        'armor_info': armor_info,
                        'light_idx': light_idx,
                        'expanded_bbox': corrector.extractor.expanded_bbox,
                        'variance_map': variance_roi,
                        'enhanced_map': enhanced_roi,
                        'bayer_roi': bayer_roi,
                        'axis': axis,
                        'top_corner': top_corner,
                        'bottom_corner': bottom_corner,
                    })

    return classified_data, variance_data

def create_dataset(image_dir, output_label_dir, output_image_dir,
                   detector, classifier, corrector=None, is_debug=False):
    """
    处理图像 -> 保存副本 -> 生成标签 + 方差图。
    """
    image_dir = Path(image_dir)
    output_label_dir = Path(output_label_dir)
    output_image_dir = Path(output_image_dir)

    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # 方差图输出目录
    variance_dir = output_image_dir / "variance_maps"
    variance_dir.mkdir(parents=True, exist_ok=True)

    image_files = (list(image_dir.glob("*.bmp")) +
                   list(image_dir.glob("*.jpg")) +
                   list(image_dir.glob("*.png")))

    for image_file in tqdm(image_files, desc="Processing Dataset"):
        image_bayer = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image_bayer is None:
            continue

        classified_data, variance_data = get_one_image_label(
            image_bayer, detector, classifier, corrector, is_debug)

        # 过滤 class_id == 6（negative）
        valid_data = [item for item in classified_data if item['class_id'] != 6]

        # 保存原始 Bayer 图像副本
        save_raw_path = output_image_dir / image_file.name
        shutil.copy2(str(image_file), str(save_raw_path))

        # 保存方差图 + 元数据
        for vdata in variance_data:
            var_map_u8 = vdata['variance_map'].astype(np.uint8)
            var_fname = f"{image_file.stem}_light{vdata['light_idx']}_variance.png"
            cv2.imwrite(str(variance_dir / var_fname), var_map_u8)

            meta = {
                'expanded_bbox': list(vdata['expanded_bbox']),
                'top_corner': (list(vdata['top_corner'])
                               if vdata['top_corner'] is not None else None),
                'bottom_corner': (list(vdata['bottom_corner'])
                                  if vdata['bottom_corner'] is not None else None),
                'axis_centroid': list(vdata['axis'].centroid),
                'axis_direction': list(vdata['axis'].direction),
            }
            meta_fname = f"{image_file.stem}_light{vdata['light_idx']}_meta.json"
            with open(variance_dir / meta_fname, 'w') as f:
                json.dump(meta, f, indent=2)

        # 保存标签文件
        label_file_path = output_label_dir / (image_file.stem + ".txt")
        label_lines = []
        if valid_data:
            for item in valid_data:
                class_id = item['class_id']
                keypoints = item['keypoints']
                color = item['color']

                line_parts = []
                for kp in keypoints:
                    line_parts.append(f"{kp[0]:.6f}")
                    line_parts.append(f"{kp[1]:.6f}")

                line_parts.append(str(class_id))
                line_parts.append(str(color))

                label_lines.append(" ".join(line_parts) + "\n")

        with open(label_file_path, 'w') as f:
            f.write("\n".join(label_lines))


if __name__ == "__main__":
    model_path = "/home/frank/RM2026/python_refactor/model/cnn.onnx"
    label_path = "/home/frank/RM2026/python_refactor/model/label.txt"
    raw_image_dir = "/home/frank/RM2026/python_refactor/dataset/competation/5-25/3/images_0525_1023/"

    detector = TraditionalArmorDetector(binary_thresh=100)
    classifier = NumberClassifier(model_path, label_path)
    corrector = LightCornerCorrector()

    create_dataset(
        image_dir=raw_image_dir,
        output_label_dir="/home/frank/RM2026/python_refactor/dataset/test",
        output_image_dir="/home/frank/RM2026/python_refactor/dataset/test",
        detector=detector,
        classifier=classifier,
        corrector=corrector,
        is_debug=True
    )