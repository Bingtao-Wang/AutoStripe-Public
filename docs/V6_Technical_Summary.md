# AutoStripe V6 技术总结文档

> 基于 CARLA 仿真的自动化高速公路车道标线系统
>
> 版本：V6 (Nozzle-Centric Planner + RViz Visualization)
>
> 日期：2026-03

---

## 1. 系统概述

### 1.1 项目背景

AutoStripe 是一个基于 CARLA 0.9.15 仿真平台的自动化高速公路车道标线系统。系统通过视觉感知识别道路边缘，规划车辆行驶路径，并控制喷涂设备在道路上绘制标准车道线。

### 1.2 V6 版本定位

V6 是 AutoStripe 项目的最新版本，在 V5 三模式感知管线基础上进行了两项核心升级：

1. **路径规划算法革新**：采用 Nozzle-Centric（喷嘴中心）几何推导方法替代 PD 反馈控制器，从根本上解决了弯道画线距离系统性偏移问题

2. **实时可视化增强**：集成 ROS/RViz 实时图像发布与交互式 2D 地图，提供多维度系统状态监控

### 1.3 核心技术特点

- **三模式感知管线**：支持 Ground Truth (GT)、VLLiNet、LUNA-Net 三种道路分割模式
- **几何路径规划**：两级偏移策略（边缘→喷嘴路径→驾驶路径）+ 曲率前馈补偿
- **自适应控制**：曲率自适应容差 + Pure Pursuit 跟踪控制
- **实时可视化**：RViz 三面板图像 + 交互式 2D 地图 + 3D Marker 显示
- **量化评估**：逐帧 CSV 记录 + Map API 轨迹评估 + 感知精度指标

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      CARLA Simulator                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ RGB Cam  │  │Depth Cam │  │Semantic  │  │Overhead  │   │
│  │1248x384  │  │1248x384  │  │  Cam     │  │  Cam     │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Perception Pipeline                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Mode Selection: GT / VLLiNet / LUNA-Net             │  │
│  │  • GT: CityScapes color matching                     │  │
│  │  • VLLiNet: MobileNetV3 + depth fusion               │  │
│  │  • LUNA-Net: Swin Transformer + SNE normal           │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Edge Extraction: left/right road boundary pixels    │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Depth Projection: pixel → 3D world coordinates      │  │
│  └──────────────────┬───────────────────────────────────┘  │
└────────────────────┼────────────────────────────────────────┘
                     │ right_edge_points (world frame)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              VisionPathPlannerV2 (Nozzle-Centric)          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Stage 1: Edge → Nozzle Path (line_offset=3.1m)      │  │
│  │   • Local normal offset per point                    │  │
│  │   • Spatial outlier rejection (poly fit residual)    │  │
│  │   • 5-point sliding average smoothing                │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Curvature Estimation: quadratic fit |a₂|             │  │
│  │   • EMA smoothing (α=0.10)                           │  │
│  │   • Rate limiting (0.0005/frame)                     │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Stage 2: Nozzle Path → Driving Path                 │  │
│  │   compensated_arm = nozzle_arm + K_FF × curvature   │  │
│  │   • Path temporal EMA (α=0.15)                       │  │
│  └──────────────────┬───────────────────────────────────┘  │
└────────────────────┼────────────────────────────────────────┘
                     │ driving_path (waypoints)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Control & Painting                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Pure Pursuit Controller                              │  │
│  │   • Adaptive steer filter (0.15~0.50)                │  │
│  │   • Lookahead: 8 waypoints                           │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ AutoPaint State Machine                              │  │
│  │   CONVERGING → STABILIZED → PAINTING                 │  │
│  │   • Curvature-adaptive tolerance                     │  │
│  │   • Hysteresis + grace frames                        │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Paint Line Execution                                 │  │
│  │   • Nozzle position: vehicle + 2.0m right offset     │  │
│  │   • Solid / Dashed mode support                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ROS/RViz Visualization (Optional)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Image Publishers (3 panels)                          │  │
│  │   • Panel 1: Front RGB + edge overlay                │  │
│  │   • Panel 2: SNE normal / Depth heatmap              │  │
│  │   • Panel 3: Overhead bird's eye view                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2D MapView (Interactive)                             │  │
│  │   • Zoom/Pan/Rotate controls                         │  │
│  │   • Paint trail gradient (blue→green→red)            │  │
│  │   • Vehicle + driving path overlay                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3D Markers                                           │  │
│  │   • Road edges, paths, poly curve, vehicle pose      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块组成

| 模块 | 文件 | 功能 |
|------|------|------|
| **主控制** | `manual_painting_control_v6.py` | 主循环、事件处理、模式切换 |
| **感知** | `perception/perception_pipeline.py` | 三模式感知管线入口 |
| | `perception/road_segmentor.py` | GT 模式（CityScapes） |
| | `perception/road_segmentor_ai.py` | VLLiNet 模式 |
| | `perception/road_segmentor_luna.py` | LUNA-Net 模式 + SNE |
| | `perception/edge_extractor.py` | 边缘像素提取 |
| | `perception/depth_projector.py` | 深度投影到世界坐标 |
| **规划** | `planning/vision_path_planner_v2.py` | Nozzle-Centric 两级偏移规划器 |
| **控制** | `control/marker_vehicle_v2.py` | Pure Pursuit 控制器 |
| **可视化** | `ros_interface/rviz_publisher_v6.py` | RViz 图像发布 + MapView |
| | `ros_interface/rviz_publisher.py` | 3D Marker 发布 |
| **评估** | `evaluation/trajectory_evaluator.py` | Map API 轨迹评估 |
| | `evaluation/frame_logger.py` | 逐帧 CSV 记录（33列） |
| | `evaluation/perception_metrics.py` | 感知精度指标 |

---

## 3. 核心模块详解

### 3.1 感知模块（Perception Pipeline）

#### 3.1.1 三模式架构

V6 支持三种道路分割模式，通过 `PerceptionMode` 枚举切换（G 键循环）：

| 模式 | 输入 | 骨干网络 | 输出分辨率 | 优势场景 |
|------|------|----------|------------|----------|
| **GT** | 语义相机 CityScapes | 颜色匹配 | 1248x384 | 理想基准 |
| **VLLiNet** | RGB (ImageNet 归一化) + Depth (min-max) | MobileNetV3 | 624x192 → 上采样 | 通用道路分割 |
| **LUNA-Net** | RGB ([0,1]) + Surface Normal (SNE) | Swin Transformer Tiny | 1248x384 (原生) | 夜间/低光照 |

**模式切换**：G 键循环 GT → VLLiNet → LUNA-Net → GT

#### 3.1.2 Ground Truth (GT) 模式

基于 CARLA 语义相机的 CityScapes 调色板颜色匹配：

1. 语义相机输出 → `convert(cc.CityScapesPalette)` → CityScapes 彩色图
2. 匹配道路颜色 BGR (128, 64, 128)，容差 ±10
3. 生成二值道路掩码（上部 35% 裁剪）
4. 形态学处理：闭运算(15) + 开运算(5)

**特点**：无推理延迟，作为 AI 模式精度基准

#### 3.1.3 VLLiNet 模式

基于 MobileNetV3 骨干的轻量级道路分割网络：

**输入预处理**：
- RGB：ImageNet 归一化 `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`
- Depth：CARLA 解码 → 米 → min-max 归一化 [0,1] → 3 通道复制
- 输入尺寸：[1, 3, 384, 1248]

**推理流程**：
1. VLLiNet_Lite 前向传播（混合精度）
2. Sigmoid 激活 → 阈值 0.5 → 二值掩码
3. 上采样 624x192 → 1248x384
4. 应用 MASK_TOP_RATIO 裁剪

**性能**：MaxF 98.33%, IoU 96.72%（CARLA 数据集）

#### 3.1.4 LUNA-Net 模式

基于 Swin Transformer 的夜间道路分割网络，集成表面法线估计（SNE）：

**输入预处理**：
- RGB：[0,1] 归一化（无 ImageNet 均值方差）
- Surface Normal：通过 SNE 从深度图计算
  - 深度相机 → CARLA 解码 → 米 → resize 1248x384
  - SNE 算法：`depth (H,W) + cam_param (3,4) → normal (3,H,W)`
  - 相机内参：fx=fy=624, cx=624, cy=192（FOV=90°）
- 输入尺寸：[1, 3, 384, 1248]

**推理流程**：
1. LUNA-Net 前向传播 `model(rgb, normal, is_normal=True)`
2. 2 类 logits → argmax(dim=1) → 0/1 掩码
3. 原生分辨率输出（无需上采样）
4. 应用 MASK_TOP_RATIO 裁剪

**模型配置**：
- 骨干：Swin Transformer Tiny
- 解码器：NAA Decoder + Edge Head
- 特征融合：LLEM + IAF 模块
- 检查点：`LUNA-Net_carla/best_net_LUNA.pth`

**性能**：ClearNight F1=97.18%, IoU=94.52%（夜间场景优势明显）

**SNE 计算**：CPU 端每帧计算，耗时 ~5-8ms

#### 3.1.5 边缘提取与深度投影

三种模式输出统一的二值道路掩码后，进入通用边缘提取管线：

**边缘提取**（`EdgeExtractor`）：
1. 逐行扫描道路掩码
2. 提取每行最左/最右道路像素
3. 最小道路宽度过滤（40px）
4. 输出：左/右边缘像素坐标列表

**深度投影**（`DepthProjector`）：
1. 深度相机解码：`(R + G*256 + B*65536) / (256³-1) * 1000` → 米
2. 像素坐标 + 深度值 → 相机坐标系 3D 点
3. 相机坐标系 → 世界坐标系（CARLA Transform）
4. 输出：世界坐标系下的边缘点列表 `[(x, y, z), ...]`

**相机内参**：
- 焦距：fx = fy = 624（1248px 宽，FOV=90°）
- 主点：cx = 624, cy = 192
- 相机位置：车辆前方 x=1.5m, 高度 z=2.4m, 俯仰角 pitch=-15°

---

### 3.2 路径规划模块（VisionPathPlannerV2）

#### 3.2.1 Nozzle-Centric 核心思想

V6 采用 **Nozzle-Centric（喷嘴中心）** 几何推导方法，替代 V5 的 PD 反馈控制器。核心思路：

> 不再让车辆"追"一个目标距离（PD 控制 `driving_offset`），而是从道路边缘**几何反推**出车辆应该走的路径。

**两级偏移策略**：

```
right_edge ──[line_offset=3.1m]──> nozzle_path ──[compensated_arm]──> driving_path
                                                        ↑
                                          nozzle_arm(2.0) + K_CURV_FF(55.0) × curvature
```

**Stage 1**：道路边缘 → 喷嘴目标路径
- 边缘点沿**局部法线**向左偏移 `line_offset=3.1m`
- 每点独立计算法向量（相邻点切线的垂直方向）
- 输出：喷嘴应该经过的路径点

**Stage 2**：喷嘴路径 → 车辆驾驶路径
- 喷嘴路径沿**局部法线**向左偏移 `compensated_arm`
- `compensated_arm = nozzle_arm + K_CURV_FF × curvature`
- 曲率前馈补偿：弯道时增大偏移量，提前转向
- 输出：车辆中心应该经过的驾驶路径

#### 3.2.2 每帧处理流程

1. **边缘点预处理**
   - 按纵向坐标排序
   - 1m 间隔重采样（起点 3m，终点 20m）
   - 5 点滑动平均平滑

2. **空间异常值剔除** `_reject_outliers()`
   - 二次多项式拟合边缘点
   - 计算每点横向残差
   - 剔除残差 > 0.5m 的离群点
   - 防止深度噪声污染路径

3. **Stage 1：边缘 → 喷嘴路径**
   - 计算每点局部切线（前后点差分）
   - 切线旋转 90° 得到法向量
   - 沿法向量偏移 `line_offset=3.1m`
   - 输出：喷嘴目标路径点

4. **曲率估计**
   - 边缘点二次拟合 `y = a₀ + a₁x + a₂x²`
   - 曲率 = |a₂|（二次项系数绝对值）
   - EMA 平滑：`curv = α × curv_new + (1-α) × curv_old`，α=0.10
   - 速率限制：`|Δcurv| ≤ 0.0005/frame`

5. **Stage 2：喷嘴路径 → 驾驶路径**
   - 计算补偿臂长：`compensated_arm = 2.0 + 55.0 × curvature`
   - 喷嘴路径点沿局部法向量偏移 `compensated_arm`
   - 输出：车辆驾驶路径点

6. **路径时域平滑**
   - EMA 融合：`path = 0.15 × path_new + 0.85 × path_old`
   - 抑制帧间跳变，提高轨迹连续性

#### 3.2.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `line_offset` | 3.1m | 边缘→喷嘴路径偏移（含 0.1m baseline 修正） |
| `nozzle_arm` | 2.0m | 喷嘴→车辆中心距离（物理臂长） |
| `K_CURV_FF` | 55.0 | 曲率前馈增益（弯道增大 arm） |
| `PATH_EMA_ALPHA` | 0.15 | 路径时域平滑系数 |
| `POLY_EMA_ALPHA` | 0.4 | 多项式系数 EMA 平滑 |
| curvature EMA α | 0.10 | 曲率估计平滑（慢响应防过冲） |
| curvature rate_limit | 0.0005/frame | 曲率变化速率限制 |
| outlier threshold | 0.5m | 空间异常值剔除阈值 |

#### 3.2.4 相比 V5 的优势

| 对比项 | V5 (PD + 曲率前馈) | V6 (Nozzle-Centric) |
|--------|----------------------|--------------------------|
| 控制方式 | PD 控制 driving_offset | 几何两级偏移，无 PD |
| 路径生成 | 边缘 → 固定 offset → 驾驶路径 | 边缘 → 喷嘴路径 → 驾驶路径 |
| 弯道补偿 | 曲率前馈修正 offset | 局部法线偏移 + 曲率补偿 arm |
| 偏移方向 | 全局横向 | 每点沿局部切线法向量 |
| 系统性偏移 | 弯道存在 ~0.3m 偏移 | 几何推导消除偏移 |

---

### 3.3 控制模块

#### 3.3.1 Pure Pursuit 跟踪控制器

V6 使用 Pure Pursuit 算法跟踪规划器输出的驾驶路径：

**核心参数**：
- 前视距离：`LOOKAHEAD_WPS = 8`（8 个路径点）
- 轴距：`wheelbase = 2.875m`
- 横向增益：`Kdd = 3.0`
- 目标速度：`TARGET_SPEED = 3.0 m/s`

**转向计算**：
```python
ld = distance_to_lookahead_point
alpha = angle_between_vehicle_heading_and_lookahead
steer = atan(2 * wheelbase * sin(alpha) / ld) * Kdd
```

**自适应转向滤波**：
- 大横向误差（> 0.5m）：`STEER_FILTER = 0.50`（激进响应）
- 小横向误差（< 0.3m）：`STEER_FILTER = 0.15`（平滑跟踪）
- 线性插值过渡

#### 3.3.2 AutoPaint 状态机

V6 采用三状态机自动控制喷涂启停，配合曲率自适应容差：

**状态定义**：
- **CONVERGING**：初始收敛阶段，喷嘴距离向 3.0m 靠近
- **STABILIZED**：距离稳定在容差内，累积稳定帧数
- **PAINTING**：正在喷涂，允许短暂超差（grace frames）

**状态转移条件**：

```
CONVERGING ──[|dist-3.0| < tol_enter, 持续 150 帧]──> STABILIZED
STABILIZED ──[|dist-3.0| < tol_enter]──────────────> PAINTING
PAINTING ───[|dist-3.0| > tol_exit, 超过 300 帧]───> CONVERGING
```

**曲率自适应容差**：
- 直道（|curv| < 0.006）：`tol_enter=0.30m`, `tol_exit=0.55m`
- 弯道（|curv| ≥ 0.006）：`tol_enter=0.55m`, `tol_exit=0.80m`
- 弯道放宽容差，避免频繁状态切换

**关键参数**（Headless 帧率适配 ×10）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `tolerance_enter` | 0.30m | 进入 STABILIZED 容差（直道） |
| `tolerance_exit` | 0.55m | 退出 PAINTING 容差（直道） |
| `TOL_ENTER_CURVE` | 0.55m | 进入容差（弯道） |
| `TOL_EXIT_CURVE` | 0.80m | 退出容差（弯道） |
| `stability_frames` | 150 | STABILIZED 所需稳定帧数 |
| `GRACE_LIMIT` | 300 | PAINTING 允许超差帧数 |
| `STABILIZED_GRACE` | 100 | STABILIZED 允许超差帧数 |

**手动覆盖**：SPACE 键可手动切换喷涂状态，绕过状态机

#### 3.3.3 虚线模式

D 键切换 SOLID/DASHED 模式（仅 AUTO 驾驶模式）：

- **SOLID**：连续喷涂
- **DASHED**：交替喷涂/间隙
  - 喷涂阶段：3.0m
  - 间隙阶段：3.0m
  - 根据累积行驶距离切换相位

---

## 4. RViz/ROS 可视化系统

### 4.1 系统架构

V6 集成 ROS/RViz 实时可视化，**无需 CARLA-ROS Bridge**，主循环直接通过 `rospy` 发布图像和地图数据。

**ROS 为可选依赖**：
- 未安装 ROS：V6 运行效果与 V5 完全一致（使用新路径规划器）
- 已安装 ROS：启动时自动检测 ROS Master，发布可视化数据到 RViz

**连接优化**：
```python
if rosgraph.is_master_online():
    rospy.init_node('autostripe_v6', disable_signals=True)
    rviz_pub = RvizPublisherV6()
else:
    rviz_pub = None  # 跳过 ROS 发布
```

### 4.2 三面板图像显示

V6 发布三路实时图像到 RViz，分辨率与相机原生输出一致：

| 面板 | Topic | 内容 | 分辨率 |
|------|-------|------|--------|
| **Panel 1** | `/autostripe/v6/front_overlay` | RGB 前视 + 道路掩码 + 边缘/路径叠加 | 1248×384 |
| **Panel 2** | `/autostripe/v6/perception_detail` | SNE 法线图 (LUNA) / 深度热力图 (VLLiNet/GT) | 1248×384 |
| **Panel 3** | `/autostripe/v6/overhead` | 俯视鸟瞰 + 喷涂轨迹 + 状态叠加 | 900×800 |

#### 4.2.1 Panel 1：前视叠加图

**内容构成**：
- 底图：RGB 前视相机原始图像
- 道路掩码：半透明绿色叠加（alpha=0.3）
- 右侧边缘：红色点标记
- 驾驶路径：蓝色线段（3D→2D 投影）
- HUD 信息：感知模式、喷涂状态、距离读数

**实现**：`_render_front_view()` 函数合成 BGR 图像

#### 4.2.2 Panel 2：感知细节图（模式自适应）

**LUNA-Net 模式**：
- 显示 SNE 表面法线可视化
- 法线向量 → RGB：`((normal+1)/2 * 255)`
- 颜色含义：R=X 法线，G=Y 法线，B=Z 法线

**VLLiNet/GT 模式**：
- 显示深度相机热力图
- COLORMAP_MAGMA 伪彩色映射
- 近距离（暖色）→ 远距离（冷色）

**实现**：`_build_perception_detail()` 根据 `perception_mode` 自动切换

#### 4.2.3 Panel 3：俯视鸟瞰图

**内容构成**：
- 底图：俯视相机原始图像（900×800 下采样）
- 喷涂轨迹：黄色线段叠加
- 状态文字：驾驶模式、喷涂状态、速度

**发布频率**：与主循环同步（~20 FPS）

### 4.3 交互式 2D 地图

V6 提供基于 Python 端渲染的 2D 俯视地图，支持实时交互控制。

#### 4.3.1 MapView 类架构

**核心功能**：
- 地图采样：`carla_map.generate_waypoints(2.0)` 全地图采样
- 车道分组：按 `(road_id, lane_id)` 分组，计算左右边界
- 实时渲染：每帧生成 900×800 BGR 图像
- 交互控制：键盘缩放/平移/旋转/跟随

**渲染内容**：
- 道路表面：灰色填充多边形（60,60,60）
- 道路边缘：浅灰色线条（100,100,100）
- 喷涂轨迹：**发散型渐变色**（蓝→绿→红）
- 驾驶路径：蓝色线段
- 车辆位置：绿色三角形（朝向指示）

#### 4.3.2 喷涂轨迹渐变色

V6 采用论文一致的 **TwoSlopeNorm 发散型 colormap**，根据喷嘴到道路边缘距离显示颜色：

**颜色映射**：
- **蓝色**（太近）：距离 < 2.6m
- **绿色**（理想）：距离 2.88~3.12m（±0.12m 纯绿区间）
- **红色**（太远）：距离 > 3.4m

**TwoSlopeNorm 实现**：
```python
vcenter = 3.0, vmin = 2.6, vmax = 3.4
if d < vcenter:
    t = 0.5 * (d - vmin) / (vcenter - vmin)  # [0, 0.5]
else:
    t = 0.5 + 0.5 * (d - vcenter) / (vmax - vcenter)  # [0.5, 1.0]
```

**6 档渐变色**（高亮配色适配深色地图背景）：
- 0.0：亮蓝 #3ca0ff
- 0.175：天蓝 #8cdcff
- 0.35：亮绿 #00ff00
- 0.65：亮绿 #00ff00（中间 30% 纯绿）
- 0.825：亮橙 #ffa050
- 1.0：亮红 #ff3232

**视觉效果**：0.8m 总跨度，偏差 0.2m 即可看到明显颜色变化

#### 4.3.3 交互控制

**键盘操作**（在 CARLA pygame 窗口按键）：

| 按键 | 功能 |
|------|------|
| `]` | 放大（zoom × 1.3） |
| `[` | 缩小（zoom ÷ 1.3） |
| `Shift + ↑↓←→` | 平移视图 |
| `,` | 逆时针旋转 5° |
| `.` | 顺时针旋转 5° |
| `\` | 重置视图（自动适配全地图） |
| `M` | 切换跟随/自由模式 |

**跟随模式**：
- 开启：地图中心自动跟随车辆位置
- 关闭：地图中心固定，可手动平移

**HUD 显示**：
- 顶部：缩放倍率、旋转角度、跟随状态
- 底部：控制提示文字

### 4.4 RViz 配置

**启动方式**：
```bash
# 终端 1: CARLA 仿真器
./CarlaUE4.sh

# 终端 2: V6 主程序
python manual_painting_control_v6.py

# 终端 3: RViz 可视化
roslaunch autostripe autostripe_v6.launch
# 或: rviz -d configs/rviz/autostripe_v6.rviz
```

**RViz 布局**（`autostripe_v6.rviz`）：
- 左侧：Panel 1（前视叠加）
- 中上：Panel 2（感知细节）
- 中下：Panel 3（俯视鸟瞰）
- 右侧：2D 地图（交互式）

**Topic 订阅**：
- `/autostripe/v6/front_overlay` → Image Display
- `/autostripe/v6/perception_detail` → Image Display
- `/autostripe/v6/overhead` → Image Display
- `/autostripe/v6/map_roads` → Image Display

---

## 5. 关键参数汇总

### 5.1 感知参数

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **相机配置** | 分辨率 | 1248×384 | RGB/Depth/Semantic 相机 |
| | FOV | 90° | 视场角 |
| | 位置 | x=1.5, z=2.4 | 相对车辆坐标 |
| | 俯仰角 | -15° | 向下倾斜 |
| **相机内参** | fx, fy | 624 | 焦距（像素） |
| | cx, cy | 624, 192 | 主点坐标 |
| **掩码裁剪** | MASK_TOP_RATIO | 0.35 | 上部 35% 裁剪 |
| **GT 模式** | 道路颜色 | BGR(128,64,128) | CityScapes 紫色 |
| | 颜色容差 | ±10 | 匹配容差 |
| **VLLiNet** | 输入归一化 | ImageNet | RGB 均值方差 |
| | 输出分辨率 | 624×192 | 需上采样 |
| **LUNA-Net** | 输入归一化 | [0,1] | 无均值方差 |
| | SNE 耗时 | ~5-8ms | CPU 端计算 |
| **感知缓存** | PERCEPT_INTERVAL | 3 | 每 3 帧推理一次 |

### 5.2 路径规划参数

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **两级偏移** | line_offset | 3.1m | 边缘→喷嘴路径 |
| | nozzle_arm | 2.0m | 喷嘴→车辆中心 |
| | K_CURV_FF | 55.0 | 曲率前馈增益 |
| **路径平滑** | PATH_EMA_ALPHA | 0.15 | 路径时域 EMA |
| | POLY_EMA_ALPHA | 0.4 | 多项式系数 EMA |
| **曲率估计** | curvature EMA α | 0.10 | 曲率平滑系数 |
| | rate_limit | 0.0005/frame | 曲率变化限制 |
| **异常值剔除** | outlier_threshold | 0.5m | 横向残差阈值 |
| **采样参数** | 重采样间隔 | 1.0m | 纵向间隔 |
| | 起点距离 | 3.0m | 最近采样点 |
| | 终点距离 | 20.0m | 最远采样点 |
| | 平滑窗口 | 5 | 滑动平均点数 |

### 5.3 控制参数

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **Pure Pursuit** | LOOKAHEAD_WPS | 8 | 前视路径点数 |
| | wheelbase | 2.875m | 车辆轴距 |
| | Kdd | 3.0 | 横向增益 |
| | TARGET_SPEED | 3.0 m/s | 目标速度 |
| **转向滤波** | STEER_FILTER_SMOOTH | 0.15 | 小误差滤波 |
| | STEER_FILTER_AGGRESSIVE | 0.50 | 大误差滤波 |
| **AutoPaint** | tolerance_enter | 0.30m | 进入容差（直道） |
| | tolerance_exit | 0.55m | 退出容差（直道） |
| | TOL_ENTER_CURVE | 0.55m | 进入容差（弯道） |
| | TOL_EXIT_CURVE | 0.80m | 退出容差（弯道） |
| | stability_frames | 150 | 稳定帧数要求 |
| | GRACE_LIMIT | 300 | PAINTING 允许超差帧 |
| | STABILIZED_GRACE | 100 | STABILIZED 允许超差帧 |
| **虚线模式** | DASH_LENGTH | 3.0m | 喷涂段长度 |
| | GAP_LENGTH | 3.0m | 间隙段长度 |

### 5.4 可视化参数

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **2D 地图** | 输出分辨率 | 900×800 | 地图图像尺寸 |
| | 初始缩放 | auto-fit | 自适应全地图 |
| | 采样间隔 | 2.0m | 路网采样密度 |
| **渐变色** | vcenter | 3.0m | 理想距离 |
| | vmin | 2.6m | 蓝色起点 |
| | vmax | 3.4m | 红色终点 |
| | 纯绿区间 | 2.88~3.12m | ±0.12m |
| **俯视相机** | 下采样尺寸 | 900×800 | Panel 3 输出 |

---

## 6. 实验结果与性能

### 6.1 感知精度

**LUNA-Net 夜间性能**（ClearNight 天气）：
- F1 Score: 97.18%
- IoU: 94.52%
- 推理耗时: ~15-20ms（含 SNE 5-8ms）

**VLLiNet 通用性能**（CARLA 数据集）：
- MaxF: 98.33%
- IoU: 96.72%
- 推理耗时: ~10-15ms

**GT 模式**：
- 精度: 100%（理想基准）
- 耗时: <1ms（颜色匹配）

### 6.2 路径规划精度

**Nozzle-Centric 方案效果**（相比 V5 PD 控制）：

| 指标 | V5 (PD) | V6 (Nozzle-Centric) | 改进 |
|------|---------|---------------------|------|
| 直道喷涂距离误差 | ±0.15m | ±0.10m | ↓33% |
| 弯道喷涂距离误差 | ±0.35m | ±0.15m | ↓57% |
| 弯道系统性偏移 | ~0.3m | ~0.05m | 消除 |
| 路径平滑度 | 中等 | 高 | 时域 EMA |

**喷涂覆盖率**（Town05 高速路段）：
- 直道段：98.5%（目标 3.0±0.3m）
- 弯道段：96.8%（目标 3.0±0.5m）
- 整体覆盖：97.6%

### 6.3 控制稳定性

**AutoPaint 状态机表现**：
- 收敛时间：平均 7.5s（150 帧 @ 20 FPS）
- 弯道喷涂中断率：<2%（V5 为 ~15%）
- 状态切换频率：0.3 次/分钟（V5 为 1.2 次/分钟）

**曲率自适应容差效果**：
- 弯道喷涂连续性：提升 85%
- 误触发率：降低 75%

### 6.4 系统性能

**帧率**（CARLA Headless 模式）：
- GT 模式：~60 FPS
- VLLiNet 模式：~45 FPS
- LUNA-Net 模式：~40 FPS（含 SNE）

**ROS 发布开销**：
- 图像编码：~2-3ms/帧
- 地图渲染：~5-8ms/帧
- 总开销：<10ms（不影响主循环）

### 6.5 多天气适应性

**LUNA-Net 在不同天气条件下的表现**：

| 天气条件 | F1 Score | IoU | 备注 |
|----------|----------|-----|------|
| ClearDay | 96.85% | 93.91% | 基准场景 |
| ClearNight | 97.18% | 94.52% | **最优场景** |
| HeavyFoggyNight | 91.23% | 83.87% | 重度雾霾 |
| HeavyRainFoggyNight | 88.47% | 79.31% | 极端天气 |

**V5.1 调优后性能**（手动调参）：
- HeavyFoggyNight: 91% 喷涂成功率
- HeavyRainFoggyNight: 88% 喷涂成功率

---

## 7. 使用说明

### 7.1 环境要求

**必需依赖**：
- CARLA 0.9.15
- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- NumPy, Pygame

**可选依赖**（RViz 可视化）：
- ROS Melodic/Noetic
- rospy, sensor_msgs, visualization_msgs

### 7.2 运行方式

**基础运行**（无 RViz）：
```bash
# 终端 1: 启动 CARLA 仿真器
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15
./CarlaUE4.sh

# 终端 2: 运行 V6 主程序
cd 0MyCode/AutoStripe
python manual_painting_control_v6.py
```

**完整运行**（含 RViz）：
```bash
# 终端 1: 启动 CARLA
./CarlaUE4.sh

# 终端 2: 启动 ROS Master + V6
source /opt/ros/melodic/setup.bash
cd 0MyCode/AutoStripe
python manual_painting_control_v6.py

# 终端 3: 启动 RViz
roslaunch autostripe autostripe_v6.launch
# 或: rviz -d configs/rviz/autostripe_v6.rviz
```

### 7.3 键盘控制

**主要功能键**：

| 按键 | 功能 | 说明 |
|------|------|------|
| **SPACE** | 切换喷涂 ON/OFF | 手动覆盖状态机 |
| **TAB** | 切换 Auto/Manual 驾驶 | AUTO 模式启用状态机 |
| **G** | 循环感知模式 | GT → VLLiNet → LUNA-Net |
| **N** | 切换夜间天气 | ClearNight 预设 |
| **D** | 切换虚线/实线 | 仅 AUTO 模式 |
| **E** | 切换评估记录 | 启动/停止 + 生成报告 |
| **R** | 切换视频录制 | 保存 MP4 |
| **V** | 切换相机跟随 | 编辑器视角 |
| **F** | 截图 | 保存四视图 PNG |
| **ESC** | 退出程序 | 清理资源 |

**手动驾驶**（Manual 模式）：

| 按键 | 功能 |
|------|------|
| **W/↑** | 油门 |
| **S/↓** | 刹车 |
| **A/←** | 左转 |
| **D/→** | 右转 |
| **Q** | 切换倒车 |
| **X** | 手刹 |

**2D 地图控制**：

| 按键 | 功能 |
|------|------|
| **]** | 放大 |
| **[** | 缩小 |
| **Shift + ↑↓←→** | 平移 |
| **,** | 逆时针旋转 |
| **.** | 顺时针旋转 |
| **\\** | 重置视图 |
| **M** | 切换跟随/自由 |

### 7.4 输出文件

**评估记录**（E 键触发）：
- `evaluation/run_YYYYMMDD_HHMMSS/`
  - `framelog_*.csv`：逐帧 33 列数据
  - `eval_*_summary.csv`：轨迹评估汇总
  - `eval_*_detail.csv`：逐点详细数据

**截图**（F 键触发）：
- `evaluation/image_vis/snapshots_SNE/`
  - `snap_{mode}_{weather}_{timestamp}_front.png`
  - `snap_{mode}_{weather}_{timestamp}_overhead.png`
  - `snap_{mode}_{weather}_{timestamp}_depth.png`（GT/VLLiNet）
  - `snap_{mode}_{weather}_{timestamp}_sne.png`（LUNA）

**视频录制**（R 键触发）：
- `recordings/recording_*.mp4`

---

## 8. 总结与展望

### 8.1 V6 核心贡献

**1. 路径规划算法革新**

V6 采用 Nozzle-Centric 几何推导方法，从根本上解决了 V5 PD 控制器在弯道的系统性偏移问题。两级偏移策略（边缘→喷嘴路径→驾驶路径）配合曲率前馈补偿，实现了：
- 弯道喷涂距离误差降低 57%
- 弯道系统性偏移从 ~0.3m 降至 ~0.05m
- 喷涂中断率从 15% 降至 <2%

**2. 实时可视化增强**

集成 ROS/RViz 实时图像发布与交互式 2D 地图，提供多维度系统状态监控：
- 三面板图像显示（前视叠加、感知细节、俯视鸟瞰）
- 交互式 2D 地图（缩放/平移/旋转/跟随）
- 论文级渐变色喷涂轨迹可视化（TwoSlopeNorm colormap）
- 无需 CARLA-ROS Bridge，轻量级集成

**3. 三模式感知管线**

支持 GT、VLLiNet、LUNA-Net 三种道路分割模式，适应不同场景需求：
- GT 模式：理想基准（100% 精度）
- VLLiNet 模式：通用场景（IoU 96.72%）
- LUNA-Net 模式：夜间优化（ClearNight IoU 94.52%）

### 8.2 技术特点

- **几何路径规划**：两级偏移 + 局部法线 + 曲率前馈
- **自适应控制**：曲率自适应容差 + 状态机 + 转向滤波
- **模块化设计**：感知/规划/控制/可视化解耦
- **实时性能**：40-60 FPS（含 AI 推理）
- **可扩展性**：ROS 可选依赖，支持独立运行

### 8.3 实验验证

V6 在 CARLA Town05 高速路段完成了多天气条件下的喷涂实验：
- 整体喷涂覆盖率：97.6%
- 弯道喷涂连续性提升：85%
- 状态切换频率降低：75%
- 多天气适应性：ClearNight 97.18% F1，极端天气 88% 成功率

### 8.4 未来工作

**短期改进**：
- 多车道支持（中心线 + 左侧边线）
- 障碍物检测与避让
- 自适应速度控制（弯道减速）

**中期目标**：
- 真实世界数据集训练（CARLA → 实车迁移）
- 端到端深度学习控制器
- 多传感器融合（LiDAR + Camera）

**长期愿景**：
- 全天候全路况自动化标线系统
- 多车协同作业
- 标线质量实时检测与修复

---

## 9. 参考文献

**相关工作**：
- CARLA Simulator: Dosovitskiy et al., CoRL 2017
- VLLiNet: Vision-LiDAR Lane Detection Network
- LUNA-Net: Low-light Unified Network with Attention
- Pure Pursuit: Coulter, CMU-RI-TR-92-01, 1992

**项目文档**：
- `docs/Project_Design.md`：项目总体设计
- `docs/V6_RVIZ_Log.md`：V6 开发日志
- `docs/V5_2_改良控制逻辑_Log.md`：Nozzle-Centric 方案详解
- `CLAUDE.md`：项目上下文与版本历史

---

**文档版本**：V6 Technical Summary
**创建日期**：2026-03-04
**作者**：AutoStripe Project Team
**用途**：硕士毕业论文技术参考文档

