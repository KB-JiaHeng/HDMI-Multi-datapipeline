# Multi-Person World Alignment: Fusing GVHMR with VideoMimic

## 1. 问题定义

GVHMR 是当前最好的单目单人 human motion estimation 系统之一，但在多人场景下有根本性限制：

- 每个人的 global trajectory 通过 velocity integration 独立计算，起点都是原点 (0, 0, 0)
- 两个并排站立的人在 GVHMR 输出中距离约 0.16m（几乎重叠）
- velocity integration 的累计误差导致两人的相对位置随时间漂移

我们的下游应用（HDMI / Isaac Lab 仿真、GMR 机器人 retargeting）需要多人在共享世界坐标系中有正确的相对位置。

## 2. 方案探索历程

### 2.1 第一次尝试：feat/multi-person_align（Camera-Space Fusion）

**思路**：GVHMR 的 `smpl_params_incam["transl"]` 包含了正确的相对位置（因为所有人在同一相机空间），只需旋转到 Y-up 即可。

**问题**：
- GVHMR 逐人裁剪估计深度，一个人蹲下时深度会膨胀（Z 偏差达 0.73m）
- 尝试用 YOLO bbox 做 human mask → body surface 污染点云 → 3D cost 无意义
- 写了 monolithic demo_multi.py → GPU OOM → 调试螺旋

**教训**：
1. 不能走捷径绕过 VideoMimic 的关键组件（SAM2、CVD）
2. 拆分 pipeline 为独立脚本避免 GPU 内存竞争
3. 先验证每个组件再集成

### 2.2 第二次尝试：feat/scene_align（VideoMimic 全面移植）

**思路**：完整移植 VideoMimic 的 scene alignment pipeline（SAM2 + CVD + MegaHunter），只替换 DROID-SLAM → DPVO。

**原则**：
- 严格复制 VideoMimic 代码，只改 import 路径
- 不修改 GVHMR 原有文件
- 每个步骤独立脚本，独立测试

## 3. Pipeline 架构

```
01_preprocess.py    → Tracking + per-person GVHMR + ViTPose-WB + DPVO
02_segment.py       → SAM2 instance segmentation
03_build_scene.py   → UniDepthV2 depth + RAFT flow + CVD → scene point cloud
04_align.py         → MegaHunter JAX optimization
05_render.py        → Multi-person rendering (incam + global view)
```

每个脚本从磁盘读取输入、写入输出，零内存共享。

### 组件替换

| VideoMimic | 我们的替换 | 原因 |
|------------|----------|------|
| DROID-SLAM | DPVO | 不需要 CUDA 11.8，rotation 质量相当 |
| VIMO (body estimation) | GVHMR | 更准确，包含 camera-space translation |
| GeoCalib (gravity) | GVHMR R_c2gv | 直接从网络预测获得，无需额外模型 |
| Multi-source depth fusion | UniDepthV2 + CVD | 更简单，依赖更少 |
| H5 文件格式 | .pt 文件 | PyTorch 生态更方便 |

## 4. 环境与依赖问题

### E1: JAX vs PyTorch cudnn 冲突
PyTorch 2.3+cu121 需要 cudnn 8.x，JAX 需要 9.x。**解决**：`os.environ["JAX_PLATFORMS"] = "cpu"`，MegaHunter 在 CPU 上 30 秒内收敛。发布时安装 `pip install jax jaxls jaxlie`（CPU-only，不安装 CUDA 插件）。

### E2: UniDepthV2 升级 PyTorch
`pip install UniDepth` 会安装 torch 2.10。**解决**：安装后重装 `torch==2.3.0+cu121`。

### E3: UniDepthV2 flash attention
`torch.backends.cuda.is_flash_attention_available()` 在 PyTorch < 2.4 不存在。**解决**：Monkey-patch。

### E4: DPVO OOM
DPVO 在干净进程中 OOM，但在 demo.py 中（ViTPose+HMR2 先跑后）正常。**解决**：per-person 按 ViTPose → HMR2 → del → DPVO 顺序执行。

### E5: RAFT utils import 冲突
RAFT 的 `from utils.utils import ...` 找到 hmr4d/utils/ 而不是 cvd_opt/core/utils/。**解决**：临时操纵 sys.path + sys.modules 缓存，以及添加 `cvd_opt/core/utils/__init__.py`。

## 5. 坐标系约定

### 5.1 各坐标系定义

| 坐标系 | 来源 | 特征 |
|--------|------|------|
| Camera frame | 原始相机 | Y-down（图像坐标），Z = depth |
| DPVO world | DPVO SLAM | 对静态相机 = camera frame |
| GVHMR global | 网络预测 | Y-up gravity-aligned，从原点开始 |
| MegaHunter world | = DPVO world | 优化在此坐标系中进行 |
| 最终输出 | G_full 旋转后 | Y-up gravity-aligned，地面 Y=0 |

### 5.2 Gravity Rotation

```python
R_c2gv = R_gv @ R_c.T  # 从 GVHMR 的 global_orient_c 和 global_orient_gv 推导
G_full = R_c2gv          # 不需要 R_any2ay
```

多帧多人平均 + SVD 投影到最近旋转矩阵。

### 5.3 R_any2ay 的取舍

最初按 VideoMimic 的惯例加了 `R_any2ay = diag(-1, -1, 1)`（Rz(π)），结果人头朝下。

实验验证：
- `R_c2gv` alone：orient local-Y = +0.996（人朝上 ✓），transl Y = -1.536
- `G_full = R_any2ay @ R_c2gv`：orient local-Y = -0.996（人朝下 ✗），transl Y = +1.536

**结论**：R_c2gv 本身已经是 Y-up 的旋转。transl Y 为负值由后处理的 Y-shift 修正（min_Y → 0）。R_any2ay 是多余的。

### 5.4 K 内参不匹配

| 来源 | 焦距 (1280×720) | 用途 |
|------|-----------------|------|
| GVHMR | f = √(w² + h²) ≈ 1468 | transl_incam 计算 |
| UniDepthV2 | f ≈ 857 | 点云构建、2D 投影 |

71% 的深度不匹配。**解决**：`bridge.rescale_transl_for_scene_K()` 将 GVHMR transl 从 K_gvhmr 重投影到 K_unidepth。

## 6. MegaHunter 移植中发现的 Bug

### 6.1 SAM2 Instance Mask `.bool()` Bug（最致命）

**现象**：MegaHunter 输出 T-pose + 零位移，cost = 524 billion。

**原因**：SAM2 mask 是 uint16 instance ID（0=bg, 1=person_1, 2=person_2）。代码做了 `.bool()`，把所有非零值转为 True，然后 `~True = False = 0.0`。每个人的关节都落在自己的 mask 区域上 → 所有人的所有关节的 3D confidence 被清零。

**级联效应**：
1. `conf = 0` for all → `valid_3d = False` → `person_frame_mask = 0`
2. `interpolate_frames` 检测到巨大 gap → 清零所有数组（包括 init_root_trans、body_pose）
3. Optimizer 收到全零输入 → 收敛到原点 + T-pose

**修复**：计算 person→SAM2 instance ID 映射，只 mask OTHER persons 的关节：
```python
msk_int = msk_val.round().long()
other_person = (msk_int > 0) & (msk_int != own_instance)
conf *= (~other_person).float().unsqueeze(1)
```

### 6.2 bbox_area 归一化缺失 + 分辨率不匹配

**现象**：修复 SAM2 bug 后，cost 从 524B 降到 135，但 2D cost (3.0) 远小于 3D cost (47.6)，比例失衡。

**原因**：
1. VideoMimic 对 ViTPose confidence 做了 `conf / bbox_area`，我们没做 → 2D cost 50000x 过大
2. 加了 bbox_area 后，我们在原始分辨率（1280×720），VideoMimic 在缩放分辨率（~512×288）。bbox_area 差异（220000 vs 50000）导致 2D cost 仍偏弱。

**修复**：将 bbox_area 归一化到 VideoMimic 的 512 max-dim 等效分辨率：
```python
resize_factor = 512.0 / max(img_max_dim, 1)
bbox_area = (bbx_size * resize_factor) ** 2
conf = conf / bbox_area
```

归一化后：3D ≈ 54，2D ≈ 61（比例平衡）。

### 6.3 数据流全面对比

对 VideoMimic 和我们的 pipeline 做了完整对比（详见 `data_flow_analysis.md`），覆盖 10 个物理量：K、cam2world、pts3d、depth、ViTPose 2D、SAM2 masks、SMPL params、init_root_trans、gravity、confidence normalization。

关键发现：除了 6.1 和 6.2 两个 bug 外，所有物理量在各自 pipeline 内部是一致的。差异来自分辨率空间不同（resized vs original），但只要 K、keypoints、point cloud 在同一分辨率空间内，就不影响正确性。

## 7. Velocity Prior（GVHMR 速度先验）

### 7.1 动机

修复上述 bug 后 MegaHunter 收敛正常，但蹲下/站立时仍有轻微 translation 漂移。原因：3D alignment cost 依赖点云深度（noisy），当人的姿态变化时深度采样位置变化 → translation 被拉动。

### 7.2 设计

GVHMR 的 global velocity（帧间位移 Δt）虽然绝对位置不对（从原点开始），但帧间运动模式非常平滑。

将 MegaHunter 的零速度先验升级为 GVHMR 速度先验：

```python
# 原来：temporal_smoothness_cost
delta = t_abs[f+1] - t_abs[f]
return delta * wt                        # 惩罚 ‖Δt‖²（零速度）

# 新增：velocity_prior_cost
delta = t_abs[f+1] - t_abs[f]
return (delta - v_target) * wt            # 惩罚 ‖Δt - Δt_gvhmr‖²（GVHMR 速度）
```

v_target = GVHMR global Δt 旋转到 MegaHunter world frame（R_align = G_full.T），scale ≈ 1.0。

### 7.3 可行性验证

- GVHMR global velocity 质量：渲染无漂移，确认 ✓
- Scale：GVHMR 和 MegaHunter 都使用 metric 单位，subsample=10 时量级接近（0.064m vs 0.071m）
- R_align：几何上 R_c2gv^T 正确（gravity→camera）。与 noisy 的 camera-space Δt 比较 cosine sim=0.66，是因为 camera-space 本身 noisy
- 代码接口：`jaxls.Cost(fn, (vars, data, weight))` 模式，新增一个 cost function + loop 即可

### 7.4 效果

velocity_prior_weight = 50 时，cost 分布：3D=49.5, 2D=36.3, velocity=21.5, smoothness=0.22。三者基本同量级，optimizer 在三者之间平衡。

## 8. 关键发现：Hybrid Output Fusion

### 8.1 问题

即使加了 velocity prior，aligned 视频中人站立时仍有可见的"晃动"（核心稳但绕着核心摆动），而 unaligned（GVHMR 原始）版本完全稳定。

### 8.2 诊断

对比 jitter 指标：

| 指标 | Aligned | Unaligned |
|------|---------|-----------|
| Translation accel | 0.000492 | 0.000936 |
| Orient angular accel | 0.029° | 0.104° |
| Body pose accel | 0.00285 | 0.00814 |

**Aligned 在所有指标上 jitter 更低！** 但视觉上 aligned 更晃。

### 8.3 根因

MegaHunter 每隔 10 帧优化 body_pose 和 global_orient 的残差旋转，然后 SLERP 插值回全帧率。问题链：
1. 每个 keyframe 得到不同的残差旋转
2. SLERP 在 keyframe 之间插值 → 产生周期约 10 帧的平滑过渡
3. 这些过渡和 GVHMR 逐帧预测不一致 → 视觉上的"摆动"

虽然统计 jitter 更低（SLERP 很平滑），但周期性摆动模式在视觉上比 GVHMR 的随机微小噪声更明显。

### 8.4 解决方案

**MegaHunter 只用于 translation，orient 和 body_pose 用 GVHMR 原始输出。**

```python
# Section 13: 输出时替换
aligned[pid]['smpl_params_global']['global_orient'] = gvhmr_original['global_orient']
aligned[pid]['smpl_params_global']['body_pose'] = gvhmr_original['body_pose']
```

**晃动完全消失**，同时保留正确的多人位置关系。

### 8.5 设计理念

| 方面 | 来源 | 理由 |
|------|------|------|
| **Translation** | MegaHunter | 点云锚定，正确的绝对位置和人间距 |
| **Global orient** | GVHMR original | 网络逐帧预测，平滑无插值伪影 |
| **Body pose** | GVHMR original | 网络逐帧预测，平滑无插值伪影 |

核心洞察：**Body estimation networks 和 scene optimizers 擅长互补的方面。** GVHMR 擅长 motion quality（逐帧预测，inherently smooth），MegaHunter 擅长 absolute positioning（点云锚定，correct inter-person distance）。

## 9. 定量结果

| 指标 | GVHMR Original | MegaHunter Only | **Hybrid (Final)** |
|------|---------------|-----------------|---------------------|
| 人间距 | ~0.16m (错误) | ~1.3m (正确) | **~1.3m (正确)** |
| 距离稳定性 (std) | 0.09m | 0.15m | **0.15m** |
| 运动平滑性 | 优秀 | 有摆动 | **优秀** |
| 绝对定位 | 错误 (原点) | 正确 (世界坐标) | **正确 (世界坐标)** |
| 站立稳定性 | 稳定 | 晃动 | **稳定** |

## 10. 输出格式与下游兼容性

输出 `aligned_results.pt`：
```python
{
    person_id: {
        "smpl_params_global": {
            "global_orient": tensor (F, 3),     # axis-angle, GVHMR original
            "body_pose": tensor (F, 63),         # axis-angle, GVHMR original
            "transl": tensor (F, 3),             # MegaHunter world-aligned
            "betas": tensor (F, 10),             # GVHMR original
        },
        "K_fullimg": tensor (F, 3, 3),
        "optimized_scale": float,
    }
}
```

与 GVHMR 原始 `hmr4d_results.pt` 格式完全兼容。GMR 的 `load_gvhmr_pred_file()` 期望 `gvhmr_pred['smpl_params_global']` 包含 `body_pose (F, 63)`, `global_orient (F, 3)`, `transl (F, 3)`, `betas (F, 10)` — 完全匹配。每个 person 的数据 `aligned_results[pid]` 即可直接使用。

## 11. 已知限制

1. **静态相机假设**：当前测试基于静态相机（DPVO 输出 identity cam2world）。运动相机理论上支持但未验证。
2. **Velocity integration drift**：GVHMR global 轨迹有累计漂移。MegaHunter 修正了绝对位置，但长视频中两人的 GVHMR orient/body_pose 可能各自漂移。
3. **人数限制**：测试了 2 人场景。更多人需要验证 SAM2 mask 质量和 MegaHunter OOM。
4. **fps**：从视频 metadata 读取（`cv2.CAP_PROP_FPS`），round 到整数存入 `tracks.pt`。

## 12. Publication 角度

**核心叙事**：GVHMR 提供了优秀的单人 motion estimation，但多人场景需要共享世界坐标。我们发现 VideoMimic 的 scene-based optimization 可以提供绝对定位，但 naively 应用会降低运动质量。通过分解输出——translation 来自 scene optimization，pose 来自 GVHMR——我们同时实现了正确的多人定位和平滑的运动质量。

**贡献点**：
1. 一个实用的单目视频多人世界坐标 motion estimation 系统
2. Body estimation network 和 scene optimizer 互补性的洞察（hybrid output fusion）
3. 移植过程中发现并修复的技术问题（SAM2 instance masking、分辨率归一化、gravity rotation）
4. GVHMR velocity prior cost 设计
