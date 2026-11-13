# 首帧 Pose 的轮廓与深度一致性检查

本文记录如何使用目标实例的 mask、RGB-D 深度和 OBJ 模型，检查 FoundationPose 注册结果是否存在明显的几何不一致。适用于单场景多个 instance 的首帧结果，例如 `results/multi_inbox03/instance*/ob_in_cam/`。

这个检查适合用于**异常筛查、排序和人工复核**。轮廓与深度匹配良好并不保证 pose 完全正确：遮挡、深度缺失、相近物体和物体对称性都可能导致错误姿态获得较好的指标。实际用于自动剔除前，应根据场景和有标注的样本校准阈值。

## 输入和坐标约定

对每个 instance，准备同一时刻的以下数据：

| 数据 | 本项目示例 | 说明 |
| --- | --- | --- |
| RGB | `demo_data/eu-plug/multi_scene_eu-plug/multi_inbox03_eu-plug/raw/rgb/<frame_id>.png` | 用于检查同帧对应关系和生成叠加图 |
| 原始深度 | `.../raw/depth/<frame_id>.png` | 与 RGB 对齐；该数据集以 `uint16` 毫米保存 |
| 相机内参 | `.../raw/cam_K.txt` | 与 RGB、深度和 mask 的分辨率一致 |
| 实例 mask | `results/multi_inbox03/instance<ID>/ob_mask.png` | 当前示例中，非零像素为该实例首帧 mask |
| OBJ 模型 | `demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj` | 必须使用估计该 pose 时的模型和原始坐标系 |
| pose | `results/multi_inbox03/instance<ID>/ob_in_cam/<frame_id>.txt` | 4×4 物体到相机的变换 |

`<frame_id>` 必须在所有输入中对应同一帧。不要直接把 `instance*/depth.png` 当作原始深度：该文件来自注册过程中的深度滤波结果。下文实测指标使用 `raw/depth/<frame_id>.png`，深度以 `depth_mm / 1000` 转为米。

已保存的 `ob_in_cam` pose 可以直接变换**原始 OBJ 顶点**到相机坐标；不要再次对 OBJ 居中或额外应用内部的中心变换。模型坐标和 pose 平移需使用同一单位，本例为米。先检查 RGB、深度和 mask 的尺寸、深度到 RGB 的对齐、相机内参、模型尺度以及帧 ID；这些不一致会使几何指标失效。

## 渲染模型

对于 OBJ 顶点 `p_obj = [x, y, z, 1]ᵀ` 和保存的 pose `T_ob_in_cam`：

```text
p_cam = T_ob_in_cam @ p_obj
u = fx * X_cam / Z_cam + cx
v = fy * Y_cam / Z_cam + cy
```

以上投影公式假设内参无 skew；一般情况使用 `K @ [X_cam, Y_cam, Z_cam]ᵀ` 后做齐次归一化。只渲染相机前方且位于视锥内的三角形。按照图像尺寸渲染**完整三角网格**，并对每个像素用 z-buffer 保留离相机最近的模型表面，得到：

- `R`：渲染得到的模型轮廓，像素值为布尔值。
- `Z_render`：模型表面在相机坐标系中的 **Z 深度**，单位为米。没有渲染到模型的像素标为无效。
- `M`：目标实例的二值 mask。
- `Z_obs`：与 mask 同分辨率、同视角的原始观测深度，单位为米；0、NaN、无穷大及无效深度不参与误差计算。

渲染时要使用透视正确插值并确认输出是相机坐标中的 Z 深度，不要将图形 API 的归一化深度缓冲值或相机到点的欧氏距离直接与深度图相减。不同渲染器在像素中心、三角形边缘和背面剔除上的约定可能略有不同，比较实验时应固定实现与参数。下文示例使用网格三角形和 z-buffer，不是顶点投影凸包。

## 指标

### 轮廓 IoU

```text
intersection = count(R ∩ M)
union        = count(R ∪ M)
silhouette_iou = intersection / union
```

值越接近 1，模型渲染轮廓与 mask 的二维重合越好。`union == 0` 时定义为**无效结果**，不要设为 0 或 1。对于 mask 非空但模型完全没有渲染到图像中的情况，IoU 为 0，应额外记录无渲染像素，检查 pose、单位和相机坐标约定。

建议同时记录两个方向的覆盖率，便于理解偏差：

```text
mask_coverage   = count(R ∩ M) / count(M)
render_coverage = count(R ∩ M) / count(R)
```

`mask_coverage` 衡量目标 mask 有多少被模型覆盖；`render_coverage` 衡量模型投影有多少落在目标 mask 内。对应分母为 0 时指标无效。注意完整模型轮廓可能包含被遮挡的部分，而 mask 可能只标注可见部分，这时模型投影超出 mask 并不一定是错 pose。

### 重合区域的深度误差

只在模型轮廓、目标 mask 和有效深度共同覆盖的像素上比较：

```text
valid = R ∩ M ∩ isfinite(Z_render) ∩ (Z_render > 0)
              ∩ isfinite(Z_obs) ∩ (Z_obs > 0)
depth_error(pixel) = abs(Z_render(pixel) - Z_obs(pixel))

depth_median_mm = median(depth_error[valid]) * 1000
depth_p90_mm    = percentile(depth_error[valid], 90) * 1000
```

`depth_median_mm` 越小表示重合区域中模型表面和观测表面越接近。中位数相对少量离群值更稳健；90 分位数有助于发现局部错位，但对遮挡和传感器噪声更敏感。若 `valid` 为空，两个指标均为**无效**；样本太少时应标记为证据不足，不能据此认定正确或错误。

可选地记录一个固定容差内的比例，用来衡量重合区域中有多少像素真正贴近模型：

```text
depth_inlier_ratio(t_mm) = count(depth_error[valid] <= t_mm / 1000) / count(valid)
```

`t_mm` 必须随相机精度、距离、物体材质和模型质量校准，并在结果中记录。还应同时输出 `mask_pixels`、`render_pixels`、`overlap_pixels`、`valid_depth_pixels` 和 `valid_depth_pixels / mask_pixels`，避免少量碰巧吻合的像素造成误判。深度误差小只说明**已重合且有效**的区域贴近，不说明整个 mask 或被遮挡面正确。

### 遮挡与深度方向

对于 `R ∩ M` 内的有效像素，还可以看有符号残差：

```text
signed_depth_mm = (Z_render - Z_obs) * 1000
```

正值表示模型表面位于观测表面之后，负值表示模型表面位于观测表面之前。正残差可能来自前景遮挡，也可能来自模型落在错误位置；负残差可能意味着模型跑到了观测表面前方，也可能来自噪声、标定偏差或模型误差。它们是诊断线索，不能单独作为真假判定。对完整模型轮廓和可见 mask 的比较，应在人工评估或自动决策时考虑被其他物体遮挡的区域。

## 建议的筛查流程

1. 验证输入同帧、内参和分辨率一致，OBJ 与估计时一致，深度和 pose 单位均正确。
2. 按 pose 渲染实例 OBJ 的轮廓和 Z 深度；输出渲染失败、空 mask、有效深度不足等数据质量状态。
3. 计算 IoU、两个覆盖率、重合区域的深度中位数、有效像素数，以及选定容差的深度内点比例。不要只保留一个综合分数。
4. 对于轮廓明显不匹配、重合区域深度明显不匹配、或有效像素不足的实例标记为待复核；根据标注样本校准后，才对明确异常执行自动剔除。
5. 保存 mask、轮廓和深度残差叠加图，结合三维点云和 OBJ 检查被遮挡部分、邻近实例及旋转方向。

推荐分别处理 `invalid`（输入或渲染无效）、`insufficient_evidence`（有效像素不足）、`suspect`（几何指标异常）和 `pass_screening`（通过筛查，不等于真值正确）。阈值和最小有效像素数需要按实际误检、漏检成本在标注数据上设定；应保留人工复核与重新注册的路径。

## multi_inbox03 首帧实测

以下使用帧 `1793960302765661576`、原始 `uint16` 深度、对应的各实例 mask、原始 OBJ 和已保存的首帧 pose。轮廓由 OBJ 三角网格逐像素深度缓冲渲染；深度误差只统计渲染轮廓、mask 和有效原始深度的交集。

| Instance | 轮廓 IoU | mask 覆盖率 | 渲染覆盖率 | 深度误差中位数 (mm) | 深度误差 P90 (mm) | 有效深度像素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.814 | 0.853 | 0.947 | 2.74 | 6.39 | 1775 |
| 2 | 0.843 | 0.920 | 0.910 | 1.26 | 6.14 | 2020 |
| 3 | 0.601 | 0.759 | 0.744 | 1.30 | 4.31 | 1426 |
| 4 | 0.710 | 0.791 | 0.875 | 3.56 | 12.15 | 2029 |
| 5 | 0.867 | 0.888 | 0.973 | 2.35 | 6.86 | 2939 |
| 6 | 0.775 | 0.801 | 0.960 | 2.79 | 7.12 | 2375 |
| 7 | 0.852 | 0.852 | 1.000 | 0.99 | 3.02 | 3427 |
| 8 | 0.741 | 0.931 | 0.784 | 2.43 | 8.89 | 2198 |
| 9 | 0.866 | 0.880 | 0.981 | 1.81 | 4.51 | 2422 |
| 10 | 0.759 | 0.886 | 0.841 | 3.45 | 9.96 | 1768 |
| 11 | 0.866 | 0.907 | 0.950 | 1.55 | 4.01 | 2465 |
| 12 | 0.823 | 0.837 | 0.980 | 2.50 | 9.60 | 2187 |
| 13 | 0.547 | 0.686 | 0.729 | 24.31 | 35.39 | 1839 |
| 14 | 0.516 | 0.722 | 0.645 | 26.56 | 33.71 | 1736 |
| 15 | 0.838 | 0.867 | 0.961 | 1.96 | 5.83 | 2265 |
| 16 | 0.688 | 0.915 | 0.735 | 5.58 | 11.18 | 1459 |
| 17 | 0.335 | 0.547 | 0.464 | 1.48 | 9.39 | 750 |

这些数字是当前场景的诊断数据，不是真值误差，也不是通用阈值。instance 13 和 14 的轮廓和深度指标均有明显偏差，值得优先复核。instance 17 的重合区域深度误差虽小，仍有很大一部分目标 mask 没有被预测模型覆盖。完整 OBJ 可能包括被遮挡的部位，所以应结合可见性和叠加图判断。

该场景中，原始 instance 13 和 17 的 `register_score.json` 分数分别约为 124.703 和 132.625，但都存在明显的几何不匹配。注册分数是网络对候选 pose 的排序信号，不能直接视作概率或替代这些几何检查。

## 实现和 eu-plug 初始阈值

`pose_geometry_validation.py` 使用 `pyrender` 离屏渲染原始 OBJ 的完整三角网格和相机 Z 深度，不依赖 FoundationPose 的网络或分数。`visualize_multi_instance_web.py` 对结果目录中每个 instance 的首帧运行检查；`--output` 同时生成独立 HTML 和同名的 `_validation.json` 报告，原始 pose 不会改变。示例：

```bash
.venv/bin/python visualize_multi_instance_web.py \
  --output results/multi_inbox06/pose_viewer.html \
  --debug-dir results/multi_inbox06 \
  --scene-dir demo_data/eu-plug/multi_scene_eu-plug/multi_inbox06_eu-plug/raw \
  --mesh-file demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj \
  --point-stride 2
```

HTML 的排序和状态仅用于安排复核，JSON 保留各项指标、路径和阈值。网页点云的采样步长不影响验证，验证使用完整分辨率的原始深度。渲染器使用 EGL、透视投影、无背面剔除；像素边缘约定不同可能导致本实现和上面的历史结果略有差异。深度量纲为毫米，模型和 pose 单位为米，内参必须为无 skew 的针孔模型。

当前 `ValidationConfig` 是 **eu-plug 初始筛查配置**，未经标注样本校准，不可当作其他 mesh 的通用阈值：

| 检查 | 告警条件 |
| --- | --- |
| Mask 有效深度 | 少于 300 点或占 mask 少于 50% |
| 重合有效深度 | 少于 200 点，不根据该小样本的深度误差判定异常 |
| 轮廓 | IoU 小于 0.50 或 mask 覆盖率小于 70% |
| 重合深度 | 绝对误差中位数大于 10 mm、P90 大于 25 mm，或 10 mm 内点比例低于 70% |
| 疑似遮挡 | 渲染区域有效观测深度中，观测表面比模型表面近超过 10 mm 的比例大于 30% |

点数按完整分辨率的有效深度像素计，每个像素对应一个可反投影点；并不能仅凭点数区分遮挡与深度缺失。疑似遮挡比例的分母是**渲染区域内有有效观测深度的像素**，深度未知比例单独报告。可见轮廓 IoU 只去掉疑似遮挡的渲染像素，未知深度仍保留在轮廓中；该指标依赖当前 pose，并非真实可见性标注。过小 mask、模型不在图内或输入不一致也应结合图像复核。

状态优先级为 `invalid`、`insufficient_evidence`、`suspect`、`pass_screening`；各状态仍保留全部独立告警，例如证据不足的 pose 也可能同时有低重合率。通过筛查不代表 pose 真实正确，遮挡、相近物体及对称性仍可造成歧义。阈值应结合人工标注结果及误检、漏检成本调整。
