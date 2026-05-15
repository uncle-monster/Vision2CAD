# Img2CAD 端到端图片转 CAD 指南

## 概述

将手机拍摄的家具照片（椅子/桌子/柜子）通过 Img2CAD 模型转换为可编辑的 CAD 模型（.obj 文件）。

**核心流程：** 输入图片 → 背景移除 → Stage 1 (Llama VLM 预测结构) → Stage 2 (GMFlow 预测参数) → 输出 .obj

## 环境要求

- GPU 显存 ≥ 24GB（Stage 1 需要加载 Llama 3.2-11B 4-bit 量化模型）
- Python 3.10+
- CUDA 11.8+

## 第一步：准备环境

```bash
# 进入项目目录
cd /root/autodl-tmp/Img2CAD

# 安装依赖
pip install -r requirements.txt

# 确认 rembg 已安装（用于背景移除）
pip install rembg==2.0.60
```

## 第二步：确认模型权重

确保以下权重文件存在（chair 类别）：

```
data/ckpts/
├── hf/
│   ├── llamaft/chair/checkpoint-2160/   # Stage 1 LoRA adapter
│   └── trassembler/chair/               # Stage 2 GMFlow checkpoint
└── trassembler/chair/
    ├── checkpoints/last.ckpt
    └── .hydra/config.yaml
```

如果缺少权重，从 HuggingFace 下载：

```bash
huggingface-cli download qq456cvb/img2cad --local-dir data/ckpts/hf

# 创建软链接使自动检测生效
mkdir -p data/ckpts/llamaft/chair
ln -s $(pwd)/data/ckpts/hf/llamaft/chair/checkpoint-2160 data/ckpts/llamaft/chair/checkpoint-2160

mkdir -p data/ckpts/trassembler/chair
cp -r data/ckpts/hf/trassembler/chair/* data/ckpts/trassembler/chair/
```

## 第三步：放置输入图片

将待转换的图片放入 `test_data/` 目录：

```bash
ls test_data/
# chair_test1.jpg
```

**图片要求：**
- 格式：JPG 或 PNG
- 内容：单件家具（椅子/桌子/柜子），物体清晰可见
- 建议：正面或 3/4 视角，光照均匀，避免强烈遮挡

## 第四步：运行推理

### 单张图片转换

```bash
python infer_single.py \
    --image test_data/chair_test1.jpg \
    --category chair \
    --text_emb_retrieval \
    --save_text
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 输入图片路径 | 必填 |
| `--category` | 物体类别：chair / table / storagefurniture | chair |
| `--output_dir` | 输出目录 | data/output/single_inference/ |
| `--no_background_removal` | 跳过背景移除（白底图可用） | 关闭 |
| `--text_emb_retrieval` | 部件名称纠错 | 关闭 |
| `--save_text` | 保存 VLM 生成的文本 | 关闭 |
| `--num_tokens` | VLM 最大 token 数 | 1024 |

### 批量转换

```bash
# 遍历 test_data 下所有图片
for img in test_data/*.jpg test_data/*.png; do
    python infer_single.py \
        --image "$img" \
        --category chair \
        --text_emb_retrieval
done
```

## 第五步：查看结果

输出文件保存在 `data/output/single_inference/<图片名>/` 目录下：

```
data/output/single_inference/chair_test1/
├── preprocessed.png      # 预处理后的图片（白底居中）
├── stage1_output.h5      # Stage 1 中间输出（离散 CAD 结构）
├── stage1_text.txt       # VLM 生成的文本描述（需 --save_text）
├── final.h5              # 最终 CAD 向量数据
├── final.obj             # ★ 可编辑的 3D CAD 模型
└── final.png             # 渲染预览图
```

### 打开 .obj 文件

- **Blender**：File → Import → Wavefront (.obj)，可直接编辑几何体
- **MeshLab**：File → Import Mesh
- **Fusion 360 / AutoCAD**：Insert → Mesh，可转换为实体
- **在线查看**：https://3dviewer.net 直接拖入 .obj 文件

## 常见问题

### 1. Stage 1 失败（无法解析 CAD 结构）

VLM 生成的文本格式不标准导致解析失败。尝试：
- 检查 `stage1_text.txt` 查看生成内容
- 重新运行（随机种子可能导致不同结果）
- 使用背景更干净、更接近白底的图片

### 2. CUDA Out of Memory

- Stage 1 需要 ~22GB 显存，Stage 2 需要 ~4GB
- 如果显存不足，Stage 2 可以用 CPU（将 `.cuda()` 改为 `.cpu()`），但会慢很多

### 3. 生成的 CAD 形状不正确

- 模型仅对 chair/table/storagefurniture 三类有效
- 手机拍照角度和训练数据差别较大时效果可能不佳
- 尝试多角度拍摄，选择最佳结果

### 4. 缺少 HuggingFace 认证

如果下载 Llama 模型失败，需设置 HF Token：

```bash
export HF_TOKEN="your_huggingface_token"
# 或者在命令中添加 --hf_token
python infer_single.py --image test_data/chair_test1.jpg --hf_token "your_token"
```

## 技术架构

```
输入图片 (test_data/chair_test1.jpg)
    │
    ▼
[背景移除] rembg → 白底居中物体
    │
    ▼
[Stage 1: LlamaFT]
    模型: Llama-3.2-11B-Vision-Instruct + LoRA
    输入: 预处理图片 + Prompt
    输出: 离散 CAD 结构（部件名、命令类型、布尔操作）
    格式: .h5 (per-part command vectors)
    │
    ▼
[Stage 2: TrAssembler]
    图像编码: DINOv2 ViT-B/14 (frozen)
    语义编码: CLIP ViT-B/32 (part names)
    核心模型: Transformer Decoder + Gaussian Mixture Flow
    扩散采样: FlowEulerODE (32 timesteps, 4 substeps)
    输出: 连续参数值 (坐标、角度、挤出距离等)
    │
    ▼
[CAD 实体生成] pythonOCC → TopoDS_Shape → STL → OBJ
    │
    ▼
输出文件 (final.obj, final.png, final.h5)
```
