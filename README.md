# RM2026 灯条角点标注工具

从比赛录屏的 Bayer RAW 图像中自动提取装甲板灯条，生成灰度 ROI 和方差图，
并输出带有角点和主轴线的预标注文件，供人工校正。

## 目录

- [环境准备](#环境准备)
- [数据提取](#数据提取)
- [输出说明](#输出说明)
- [人工标注流程](#人工标注流程)
- [查看结果](#查看结果)
- [文件结构](#文件结构)

## 环境准备

```bash
pip install opencv-python numpy tqdm onnxruntime
```

确保 `model/` 目录下有以下文件：

| 文件 | 用途 |
|------|------|
| `cnn.onnx` | 数字分类模型，过滤负样本（class 6） |
| `label.txt` | 分类标签列表 |
| `yolo.onnx` / `yolo.bin` / `yolo.xml` | YOLO 检测模型（备用） |

## 数据提取

### 命令行

```bash
cd src
python extract_light_data.py \
    --dataset_dir ../dataset/competation \
    --out_dir ../test/light_data \
    --model ../model/cnn.onnx \
    --label_txt ../model/label.txt
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_dir` | `../dataset/competation` | 原始 Bayer `.bmp` 图像目录（递归搜索） |
| `--out_dir` | `../test/light_data` | 输出目录 |
| `--model` | `../model/cnn.onnx` | ONNX 分类模型路径 |
| `--label_txt` | `../model/label.txt` | 分类标签文件 |

### 处理流程

1. **检测**：在 Bayer RAW 图像上用传统方法检测灯条，配对为装甲板
2. **过滤**：用 CNN 分类器剔除 class 6（负样本 / 误检）
3. **角点重提取**：对每个灯条 ——
   - 沿灯条方向扩展 ROI
   - 加权 PCA 拟合对称主轴
   - Bayer 通道比率优化 → 方差图
   - 沿方差图扫描，定位最大亮度跳变处作为角点
4. **输出**：灰度 ROI 图、方差图、标注文件

## 输出说明

```
{out_dir}/
├── gray_roi/       # 灯条灰度 ROI 原图 (.png)
├── variance_map/   # 方差图 (.png) —— 边缘/纹理突出，便于辨认同名角点
└── labels/         # 标注文件 (.txt + .json)
```

### 文件命名

```
{原图名}_armor{装甲板编号}_light{灯条编号}.png
{原图名}_armor{装甲板编号}_light{灯条编号}.txt
{原图名}_armor{装甲板编号}_light{灯条编号}.json
```

- `armor0`、`armor1`…… 同一原图中的不同装甲板
- `light0`、`light1` 同一装甲板的左右两个灯条

### 标注格式

#### .txt 文件

```
<top_x> <top_y> <bottom_x> <bottom_y>
<axis_p1_x> <axis_p1_y> <axis_p2_x> <axis_p2_y>
```

- 第 1 行：角点坐标（top、bottom），值域 [0, 1]，相对于 ROI 图像归一化
- 第 2 行：主轴线段端点，值域 [0, 1]，同样归一化
- 若角点检测失败，第 1 行为空行

#### .json 文件（LabelMe 格式）

包含 3 个 shape：

| label | shape_type | 说明 |
|-------|------------|------|
| `top` | point | 灯条上端点 |
| `bottom` | point | 灯条下端点 |
| `axis` | line | 主轴线段（沿 PCA 方向，覆盖灯条全长 + 20%） |

坐标均为 ROI 内像素坐标。

## 人工标注流程

提取完成后，角点是自动检测的，**需要人工校正**。推荐使用 LabelMe：

### 1. 安装 LabelMe

```bash
pip install labelme
```

### 2. 打开标注

```bash
labelme {out_dir}/gray_roi/   # 或 variance_map/
```

LabelMe 会自动识别同目录下的 `.json` 文件，显示预标注的角点和主轴线段。

### 3. 校正方法

- **沿主轴线段移动端点**：绿色 `axis` 线表示灯条对称主轴方向，top/bottom 端点应沿此线放置
- 方差图中灯条两端有明显亮度跳变（暗→亮），是角点的判断依据
- 若看不清楚，切换到 `gray_roi/` 下的原图对照
- 修正后保存，LabelMe 会覆盖 `.json` 文件

### 4. 标注规范

- 角点位于灯条的**端点中心**（宽度方向中间、长度方向末端）
- 若灯条被遮挡或超出图像边界，标注可见部分的端点
- 方差图比灰度图更容易定位端点，建议优先在方差图上标注

## 查看结果

可用内置查看器快速浏览提取结果和角点质量：

```bash
cd src
python view_light_data.py \
    --data_dir ../test/light_data \
    --dataset_dir ../dataset/competation
```

交互操作：

| 按键 | 功能 |
|------|------|
| `n` | 下一张 |
| `p` | 上一张 |
| `q` / `Esc` | 退出 |

显示布局：

```
┌─────────────────────────────┐
│  原图（含 expanded_bbox）    │
│  green=bbox  red=corners    │
├──────────────┬──────────────┤
│  gray ROI    │  variance    │
│  + 角点标注   │  + 角点标注   │
└──────────────┴──────────────┘
```

## 文件结构

```
python_refactor/
├── model/
│   ├── cnn.onnx          # CNN 数字分类模型
│   ├── label.txt         # 分类标签
│   ├── yolo.onnx         # YOLO 检测模型（备用）
│   └── ...
├── src/
│   ├── extract_light_data.py   # 主提取脚本
│   ├── light_corner_corrector.py  # 角点重提取算法
│   ├── lable_generator.py      # 灯条检测 & 数字分类
│   ├── view_light_data.py      # 交互式查看器
│   └── cp_image_lable.py       # 图片/标签复制工具
├── dataset/
│   └── competation/      # 原始 Bayer .bmp 图像
└── test/
    └── light_data/       # 提取输出（示例）
```
