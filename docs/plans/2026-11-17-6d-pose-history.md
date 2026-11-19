# 6d_pose_history 实施计划

日期：2026-11-17  
状态：已实施并部署；用法见 [6D 位姿历史](../6d_pose_history.md)。  
需求补充：线上 FoundationPose 推理固定使用 `debug=2`。

## 1. 目标与实现决策

在 FoundationPose Docker 内提供 `6d_pose_history`：自动保存每次单帧注册的输入、输出和 debug 产物，按时间倒序浏览历史，点击记录后查看当次 RGB-D 点云、估计位姿和模型。

确定采用以下方案：

- **归属**：推理服务、历史存储、HTTP API、历史列表与详情页面代码都放在 FoundationPose 仓库，运行于现有 FoundationPose GPU Docker。`realman_pi` 负责采集、调用服务、页面入口和同源代理。
- **线上 debug**：线上 `register()` 固定 `debug=2`；每次请求使用独立 debug 目录。历史功能启用时不提供回退到 `debug=0` 的正常运行路径。
- **记录粒度**：一次实际执行的 `register()` 对应一条历史。同一 observation 更换 mask、选择不同实例或重新解算，分别产生新记录；不以 observation ID 覆盖旧结果。
- **主键**：服务端受理推理时分配的 UTC 纳秒时间，SQLite `INTEGER PRIMARY KEY`。相机拍摄时间单独保存，不作为唯一键。
- **列表预览**：并排显示当次原始 `RGB.png` 和 `vis_score.png` 的最高分完整首行；保留四个对比图及其分数文字。
- **详情**：参照 `results/multi_inbox08/pose_viewer.html`，复用其源码和点云格式，改成按历史 ID 加载归档。
- **持久化**：SQLite 索引 + 文件目录，挂载到宿主机；重启、重建容器后仍能回看。

首版覆盖单帧 `register()` 历史，不记录后续 `track_one()` 视频序列。历史页面提供查看和下载，不接入机器人运动流程。

## 2. 现有实现与约束

| 位置 | 已确认的行为 | 对本功能的影响 |
| --- | --- | --- |
| `estimater.py:FoundationPose.register()` | `debug>=2` 输出 `color.png`、`depth.png`、mask、点云、`vis_refiner.png`、`vis_score.png`；返回原始 OBJ 坐标到相机坐标的变换 | 直接利用现有 debug 产物；不能再次居中模型 |
| `learning/training/predict_score.py:ScorePredictor.predict()` | 按 `scores.argsort(descending=True)` 生成 `vis_score` | 第一行就是最高分候选 |
| 同文件 `vis_batch_data_scores()` | 每行缩放到 100 px 高，默认追加 5 px 白色分隔；每行四列对比图 | 不能将整张长图缩成缩略图，也不能按图片高度或候选数猜测裁剪边界 |
| `results/multi_inbox08/instance1/vis_score.png` | 实际为 409 × 26460；首行内容 409 × 100 | 可作为首行预览验收样本 |
| `visualize_multi_instance_web.py`、`web/multi_pose_viewer/` | RGB-D 反投影、二进制点云、Plotly 页面、mask 选择、OBJ/坐标轴、几何指标、单文件导出 | 应复用源码；参考 HTML 是约 6.6 MB 的导出产物，不复制为业务模板 |
| `results/multi_inbox08/pose_viewer.html` | 640 × 480、同一首帧、19 个实例 | 用于验证界面与几何表现；线上单次注册详情可以只有一个实例 |
| `realman_pi/src/pose6d/pose6d/runtime.py` | 当前线上 wrapper 创建 `debug=0` 的 estimator，共用一个 `TemporaryDirectory` | 需要迁移服务归属并改为 `debug=2`、按请求输出，不能只修改一个数字 |
| `realman_pi/src/pose6d/pose6d/server.py` | WebSocket/MessagePack、8092、单 GPU worker；超时或取消后 worker 可能继续运行 | 归档必须由实际 worker 完成，不能只在 WebSocket 返回成功时保存 |
| `realman_pi/src/pose6d/pose6d/observation.py` | 已保存拍摄 timestamp 和 camera frame，但 `request()` 目前只发送 RGB-D、mask、K 和身份字段 | 扩展可选元数据字段；保留已有请求兼容性 |
| `realman_pi/config/docker/compose.pose6d.yaml` | 根文件系统只读，只有 `/tmp` 为临时写目录；没有历史 volume | 必须新增专用可写挂载，不能依赖 `/tmp` 持久化 |
| 当前 Pose Dockerfile | 从 FoundationPose 仅复制部分算法目录，服务代码/依赖来自 `realman_pi`；当前服务依赖列表未包含 Plotly | 构建时补齐本仓库服务、页面和依赖，保持原推理协议 |

另外，`register()` 在保存 debug 深度前已经做了腐蚀和双边滤波。其 `depth.png`、`scene_complete.ply` 不能代替原始输入；历史默认点云应像参考查看器一样来自原始 RGB-D。

## 3. 服务与仓库边界

```text
realman_pi
  采集/冻结 RGB-D + 选择 mask
       │ 现有 WebSocket/MessagePack register，端口 8092
       ▼
FoundationPose Docker
  pose_service：单 GPU worker、debug=2、筛查、响应
       │
       ├─ pose_history：时间主键、输入快照、debug 产物、SQLite
       └─ HTTP：列表/API/查看器，端口 8095
                 ▲
                 │ 固定目标的同源 HTTP 代理
realman_pi Web：/6d_pose_history/ → FoundationPose HTTP
```

FoundationPose 侧建议目录：

```text
pose_service/
  __init__.py
  server.py                  # 现有 WS 生命周期 + HTTP 服务启动
  runtime.py                 # FoundationPose wrapper，固定 debug=2
  protocol.py                # 兼容既有 register 请求及新增可选元数据
  wire.py                    # 与既有 MessagePack ndarray envelope 兼容
  screening.py               # 现有线上筛查逻辑
pose_history/
  __init__.py
  store.py                   # SQLite schema、时间主键、状态与恢复
  recorder.py                # 输入/输出快照、归档发布
  preview.py                 # vis_score 首行预览
  viewer_data.py             # 从历史快照提供查看器数据
  http.py                    # 只读 HTTP API 与页面
  import_legacy.py            # 显式导入旧结果
web/pose_history/
  index.html
  history.js
  style.css
web/multi_pose_viewer/        # 复用并适配详情源码
docker/
  pose6d-server.Dockerfile
  pose6d-requirements.txt
tests/
  test_pose_history_store.py
  test_pose_history_preview.py
  test_pose_history_service.py
  test_pose_history_viewer.py
```

将现有 `realman_pi` 中算法端的 server/runtime/请求校验/筛查迁入 `pose_service`，保留协议与既有筛查行为，必要的刚体矩阵校验一并收进本仓库。服务不导入 ROS、机器人控制模块或 `realman_pi` Python 包。`wire.py` 保持 ndarray 编码、dtype 白名单和 32 MiB 消息上限，使用双方互通测试防止协议分叉。

`realman_pi` 保留 observation、ROS bridge、机器人逻辑及客户端；迁移后不维护第二份 GPU 服务实现。部署切换到本仓库 Dockerfile/入口，仍由既有启动脚本管理同一个 `pose6d_server` 服务。不要把整个 `src/pose6d` 包搬走。

不新建名为 `foundationpose/` 的 Python 源码目录：本仓库 `.gitignore` 当前忽略该目录，容易漏提交。

## 4. 时间主键、索引与记录身份

### 4.1 时间语义

- `history_time_ns`：历史主键，服务端受理推理的 UTC 时间，纳秒整数。
- `received_at_ns`：实际服务器墙钟时间，不做单调修正。
- `capture_timestamp_ns`：客户端传入的拍摄时间，可空；另存 `capture_clock`，例如 ROS system time，避免将仿真时钟误显示为真实日期。
- `started_at_ns`、`finished_at_ns`：实际计算时间；耗时使用单调时钟计量。
- 页面默认按 `history_time_ns DESC` 排列，时间显示为 `Asia/Shanghai`，可查看 UTC 和原始 ID。

在 SQLite `BEGIN IMMEDIATE` 事务内，读取持久化高水位并分配 `max(time.time_ns(), last_time_ns + 1)`；与新记录一起提交。这样并发受理、同纳秒、时钟回拨和服务重启均不会覆盖历史。高水位独立保存，未来清理历史时也不能重置。另存真实墙钟值，使主键的单调修正可追溯。

**API/JSON/URL 中所有纳秒时间都作为十进制字符串传输**，避免浏览器 `Number` 超过 `2^53` 丢精度。前端排序直接依赖服务端顺序，不将 ID 先转为 `Number`。

### 4.2 SQLite 结构

主表 `pose_history` 至少包括：

| 字段 | 作用 |
| --- | --- |
| `history_time_ns INTEGER PRIMARY KEY` | 唯一历史 ID |
| `request_id TEXT UNIQUE`、`observation_id TEXT` | 关联调用；重发同一请求不重复执行 |
| `input_sha256 TEXT` | RGB、原始 depth、mask、K、模型身份的规范化摘要 |
| `object_id`、`model_sha256`、`model_asset_id` | 绑定当次模型与几何/材质快照 |
| `instance_id`、`mask_source` | SAM 实例或手工 mask 来源，可空 |
| 时间、camera frame、图像尺寸 | 检索与来源说明 |
| `status`、`score`、`screening_status` | 推理状态、原始分数与当次筛查状态 |
| `archive_status`、`preview_status`、`error_code` | 区分算法结果与归档/预览故障 |
| `record_relpath`、`schema_version` | 可搬迁目录与数据版本 |

为 observation、capture time、object/status 和时间建立必要索引；启用 WAL、短事务和 busy timeout，每个线程使用自己的连接。

同一 `request_id` 重发时，摘要相同则返回已有状态/结果；仍运行则返回已有 history ID 及 busy 状态；摘要不同则拒绝。重新点击解算产生新 `request_id`，即使 RGB-D 相同也保留为新一次历史。

首版默认一条记录显示该次注册的一个实例。后续可按 observation 聚合查看多次实例结果，但只有 RGB、原始深度、K、camera frame 均一致时才允许叠加；不能把相邻时间的不同拍摄拼成多实例场景。

## 5. 归档内容与持久化流程

### 5.1 文件布局

```text
/data/pose_history/
  history.sqlite3
  models/<model_asset_id>/        # 不可变模型包；按内容摘要去重
    model.obj                    # 推理使用的原始坐标/米制几何
    ...                          # 引用的 MTL/纹理（如有）
    manifest.json
  records/YYYY/MM/DD/<history_time_ns>/
    manifest.json
    RGB.png                      # 原始输入 RGB，无损、原分辨率
    depth_m.npy                  # 原始 float32 米制输入，无损
    cam_K.txt
    mask.png                     # 此次实际使用的二值 mask
    pose.txt                     # register 返回的原始 OBJ → 相机 4×4
    result.json                  # 分数、筛查、参数、耗时与结果身份
    preview/
      rgb_thumb.jpg
      vis_score_top1.png         # 完整最高分首行
    debug/
      color.png
      depth.png                  # 算法滤波后毫米 PNG
      ob_mask.png
      scene_raw.ply
      scene_complete.ply
      init_center.ply
      vis_refiner.png
      vis_score.png
    cache/                       # 可删除、可重建的展示产物
      scene_stride2.bin
      mask_preview.jpg
      masks.png
```

日期分区使用 UTC，与主键保持一致；页面转换为本地日期。`manifest.json` 保存相对路径、文件摘要、输入/深度单位、坐标约定、代码/模型/权重版本、迭代次数、`debug=2`、时间字段、参数和筛查阈值。没有版本信息的旧档案明确标记为未知。

原始 `RGB.png` 在进入 `register()` 前保存；OpenCV 编码必须做 RGB→BGR，不能引入颜色互换。原始深度用 `.npy` 保留 float32，读取禁用 pickle，不因毫米 PNG 量化改变点云。debug 的 `color.png` 可以与输入做像素一致性检查，但保留原有 debug 命名。

模型快照至少保存推理时加载的原始顶点/面和单位；若保存 OBJ 文件，连同其引用资源组成不可变模型包并校验。不能只记录宿主机绝对路径或在回看时加载“当前模型”。原始 OBJ SHA-256 继续用于现有协议身份，`model_asset_id` 标识完整归档包，避免材质变化混淆。

### 5.2 推理生命周期

1. 校验请求与模型身份、去重后分配时间主键，创建 `running` 索引行和独立 staging 目录，先保存原始 RGB-D、K、mask 及上下文。
2. 单 GPU worker 为该请求设置 `estimator.debug=2`、`estimator.debug_dir=<staging>/debug`，创建目录后执行 `register()`。不复用其他请求的 debug 路径；estimator 只由该 worker 访问。
3. 校验 pose、有限分数及 `best_score`，保存原始坐标下的变换、原始 scorer 分数和当次筛查结果。稀疏深度导致提前返回、未生成 scored pose 时按失败记录，不把猜测平移标记为成功。
4. 生成首行预览，校验必需文件、摘要和 JSON，写好最终 manifest；所有已保留的 debug=2 文件一起归档。
5. 文件 flush/fsync 后，在同一持久化文件系统内将 staging 原子 rename 到最终目录，再提交索引的完成状态；响应带 history ID 和归档状态。
6. 浏览列表可显示运行/失败记录，只有已发布且校验通过的文件可下载；`running` 不暴露半写入图片。

文件系统与 SQLite 无法组成一个事务，需定义恢复规则：启动时扫描未终结索引和 staging；最终目录已发布但数据库仍为 running 时按 manifest 校验恢复，未发布且 worker 已不存在时标记 `interrupted`。孤立目录进入恢复清单，不静默删除。

推理失败仍保留已保存的 RGB 和可用 debug 产物；没有 `vis_score` 时显示具体状态占位。HTTP 取消、WebSocket 断开或客户端超时，只改变交付状态，不取消正在执行的 GPU 工作和归档；实际 worker 完成后仍保存结果。归档接在 worker 的 `try/finally` 流程中，不能依赖连接仍然存在。

推理成功而首行预览生成失败时，保留 pose 并标记 `preview_status=error`，不能误显示其他请求图片。归档写入失败单独报告 `archive_status=failed`，不能声明已保存。启动时历史目录不可写则 readiness 失败；运行时磁盘不足，已算出的 pose 仍按既有协议返回并附归档错误，暂停接受新的历史推理，避免持续产生无法回看的结果。

保留完整 `vis_score`/`vis_refiner` 原图，首版不自动删除历史。提供记录数、占用和磁盘剩余量监测，后续清理策略再单独实现。debug=2 产生的 CPU 拷贝、长图编码和磁盘耗时必须进入性能验收。

## 6. 最高分预览的准确实现

当前 `vis_score` 每行是：`rgbA | depthA | rgbB | depthB`，保留现有图像顺序和分数标注。保存首行全宽，去掉下一行前的白色分隔，不单截某一列或用最终 overlay 代替。

实施方式：

1. 将行高 `100` 和间距 `5` 提取为可复用的布局定义，现有渲染函数和裁剪工具使用同一来源，避免历史模块长期硬编码另一套布局。
2. 当前格式首行裁剪为 `vis_score[0:100, 0:width]`；样本应得到 **409 × 100**，像素与原文件对应区域完全一致。
3. 保存并核对 top candidate ID、最高分及首行边界。分数相同时统一可视化与最终 pose 的候选排序，使首行与返回的 best pose 对应。
4. 列表使用 `object-fit: contain` 显示整行；RGB 可缩略，但点击后查看原始 `RGB.png`。完整 `vis_score.png` 在详情中按需打开/下载。
5. 某次失败未产出 score 图时显示“未生成评分图”；不能回退到目录里其他帧或其他请求的图片。

分数保存为原始 scorer 值。当前实现包含 pairwise selection offset（`+100`），不能当作概率或套用 0–1 进度条；列表默认按时间排列。

## 7. 页面与 API

### 7.1 历史列表

入口为 `/6d_pose_history/`。每条卡片/行展示：

- 上海时间（至少显示毫秒，完整纳秒 ID 可复制）、对象/实例、状态和原始 score。
- 左侧当次 `RGB.png` 缩略图，右侧最高分首行，二者明确标注。
- 点击进入 `/6d_pose_history/<history_time_ns>/`；浏览器后退保留筛选和翻页位置。

提供时间范围、object、状态筛选，默认每页 30 条、上限 100，使用时间 cursor 分页。列表 API 不加载点云、不返回整张 vis_score 或全分辨率 RGB；图片懒加载。运行中、失败、无记录、服务不可达和预览缺失均有独立反馈。

### 7.2 点云详情

复用 `web/multi_pose_viewer/` 的布局：左侧 RGB/mask 和实例列表，右侧点云及 pose/OBJ/坐标轴，底部位姿与筛查指标；增加返回历史按钮、记录时间、RGB 与评分首行查看区。

保留参考界面的核心交互：旋转/平移/缩放、正视/侧视/聚焦、完整/物体附近点云切换、点大小、OBJ 开关及透明度、坐标轴长度、mask 放大与实例选择。单实例历史也使用同一数据模型；字符串 instance ID 通过内部数值 `viewer_id` 映射到 mask map，页面仍显示真实 ID。

把当前 `viewer.js` 写死的 `/api/info`、`/api/scene` 等地址改为可配置的记录数据地址；CSS/JS/Plotly 资源使用支持 `/6d_pose_history/` 前缀的路径。原有命令行查看器和单文件导出继续可用。

点云从该记录的 `RGB.png + depth_m.npy + K` 重建；默认 stride=2，采样后仍使用原图像素坐标。对有限正深度，以 `ray=K^-1·[u,v,1]`、`p=ray·z/ray_z` 反投影，兼容已被线上协议接受的内参形式，不把参考脚本对无 skew 的限制无意带入线上。

坐标为 RGB optical camera frame，X 右、Y 下、Z 前，单位米；RGB 逐点配色。模型用原始 OBJ 顶点乘当次 `T_camera_object`，不再次居中、缩放或叠加 base/tool 变换。默认不采用滤波后的 debug PLY 作为场景来源。

复用现有点云二进制格式：`uint32_le point_count + float32_le[N,3] xyz + uint8[N,3] rgb`。点云和模型按需获取并缓存，CPU/文件任务放独立有界线程池，不占 GPU worker 或阻塞 asyncio loop。缓存键包含输入摘要、stride 和转换版本；原始档案始终可重建缓存。

默认展示当次已归档的 screening/阈值，不在打开历史时用当前参数重新计算并覆盖结论。适配查看器的字段与指标名称；若以后提供重新筛查，应另列版本和时间。

### 7.3 HTTP 契约

以下为 FoundationPose HTTP 服务的原生路由；代理完整保留该前缀。

| 方法与路径 | 内容 |
| --- | --- |
| `GET /healthz` | HTTP、索引、写目录、GPU 就绪/忙状态 |
| `GET /6d_pose_history/` | 历史列表页面 |
| `GET /6d_pose_history/<id>/` | 详情页面 |
| `GET /6d_pose_history/assets/<asset>` | 本地 JS/CSS/Plotly，无 CDN |
| `GET /6d_pose_history/api/records?limit=30&before=<id>&from=<ns>&to=<ns>&object_id=...&status=...` | 分页摘要及下一 cursor；时间范围 `[from,to)` |
| `GET /6d_pose_history/api/records/<id>` | manifest 摘要、pose、筛查、上下文及产物可用性 |
| `GET /6d_pose_history/api/records/<id>/preview/rgb` | 小 RGB 预览 |
| `GET /6d_pose_history/api/records/<id>/preview/score` | 最高分首行 PNG |
| `GET /6d_pose_history/api/records/<id>/viewer/info` | 查看器实例/位姿/归档筛查数据 |
| `GET /6d_pose_history/api/records/<id>/viewer/scene?stride=2` | 二进制点云 |
| `GET /6d_pose_history/api/records/<id>/viewer/model` | 快照模型顶点/面 |
| `GET /6d_pose_history/api/records/<id>/viewer/mask-preview` | RGB/mask 标注预览 |
| `GET /6d_pose_history/api/records/<id>/viewer/mask-map` | 数值 viewer ID 图 |
| `GET /6d_pose_history/api/records/<id>/files/<artifact>` | 按白名单下载 RGB、score 原图、pose 等 |

所有文件地址由记录 manifest 和产物白名单生成，限制在历史根目录内，不接受任意绝对路径、`../` 或外部 URL。记录不存在返回 404；仍在生成的必需产物返回带状态的 409，页面据此轮询；记录损坏返回明确错误。

现有 WS `result` 增加可选字段：

```json
{
  "history_id": "1794873600000000000",
  "history_path": "/6d_pose_history/1794873600000000000/",
  "archive_status": "complete"
}
```

既有 `request_id`、`observation_id`、pose、score 和 screening 不改变。`history_path` 是相对站点路径，不返回 Docker 内部地址。请求扩展可选 `capture_timestamp_ns`、`capture_clock`、`camera_frame`、`instance_id`、`mask_source`；旧客户端缺少时标记 unknown/null，不能猜测来源。

## 8. Docker 与 realman_pi 接入

FoundationPose 镜像增加 `pose_service`、`pose_history`、Web 静态资源和已锁定的 HTTP/MessagePack/Plotly 依赖；新增 `aiohttp` 承载 HTTP，WS 保持现有 8092。单进程并行运行 HTTP 与 WS，GPU 仍只有一个专用 worker，回看不会重新加载模型或执行推理。

部署参数建议：

```text
POSE6D_HISTORY_ROOT=/data/pose_history
POSE6D_HISTORY_HTTP_PORT=8095
POSE6D_HISTORY_DIR=<宿主机持久化目录>
```

Compose 新增 `${POSE6D_HISTORY_DIR:-../FoundationPose/data/pose_history}:/data/pose_history:rw`，8095 默认只映射宿主机 loopback；根文件系统继续只读。staging 与最终档案必须同在该 volume，避免跨文件系统 rename。`/tmp` 只存临时缓存，不放历史；启动检查挂载权限和可用空间。

`realman_pi` 侧仅调整：

1. `ObservationStore.request()` 传递拍摄时间/时钟、frame、所选实例及 mask 来源；bridge 将 `history_id/history_path/archive_status` 透传到现有结果事件。
2. 在主页面和 `/6d_pose.html` 增加“6D 历史”，解算后增加“查看本次历史”。入口指向 `/6d_pose_history/`，列表和详情实现不复制进 `realman_pi`。
3. 在现有 aiohttp `web_server.py` 的静态兜底路由之前注册 `/6d_pose_history/` 及其子路径代理，目标为配置中的固定 FoundationPose HTTP 地址。流式转发图片/点云、设置超时，服务不可达时返回可读的 502。Web 控制容器访问宿主机服务沿用现有网络连接方式，不把它自身的 `127.0.0.1` 当作算法宿主机。
4. 更新 Compose 构建来源、入口、持久化挂载和端口配置；`rm65` 继续管理原服务名，历史随算法服务启动。浏览器通过现有机器人 Web 地址访问，不依赖跨域、另开公网端口或 GPU 主机地址硬编码。

直接访问 FoundationPose HTTP 也能独立浏览历史，证明数据和界面功能归属于本仓库。

## 9. 旧结果导入

提供显式 CLI 导入已有结果，先 dry-run 展示可用记录与缺失项，再执行导入。新线上请求自动归档；启动时不扫描整个 `results/` 猜测历史。

以 `results/multi_inbox08` 为首个导入样本，读取其 batch summary、各实例首帧 pose、`register_score.json`、debug score 图，以及原始场景 RGB/depth/K 和原模型。按真实首帧文件名匹配 RGB-D；19 个实例分别生成记录，用同一 observation/group 元数据关联，不覆盖同帧结果。

导入时间分配新的历史主键，原帧 timestamp 单独保留为 capture/source 时间，记录 `source=legacy_import`。对 legacy 唯一来源键建立导入去重，重复执行不重复生成。导入复制完整快照，不长期引用源绝对路径；缺少 RGB-D/K/模型时保留缺失说明，不能伪造可用点云或用当前资源补齐。多实例合并回看作为后续扩展，首版重点验证每条实例结果均可正确查看。

## 10. 分阶段实施与验收

### 阶段 1：算法服务归属与 debug=2

- [x] 迁移线上算法服务到本仓库，保持 WS/MessagePack 和模型校验兼容；只迁移服务端必需代码。
- [x] 固定线上 `debug=2`，逐请求隔离 debug 目录，保留单 GPU worker。
- [x] 更新 Docker 构建与依赖，完成旧客户端到新服务的真实注册验证。

### 阶段 2：记录与持久化

- [x] 实现 SQLite schema、时间高水位、请求去重、输入/模型快照、manifest 和归档发布。
- [x] 覆盖完成、失败、提前返回、取消/断线后完成、启动恢复与磁盘写入失败。
- [x] 挂载历史目录，验证重建容器仍能查询和读取旧记录。

### 阶段 3：预览和历史 API

- [x] 实现 score 行布局共享、首行裁剪、RGB 缩略图、分页和时间筛选。
- [x] 实现白名单产物路由、归档状态、只读 HTTP 及资源缓存。
- [x] 确认卡片加载不会读取整个评分长图或构造点云。

### 阶段 4：点云详情与界面复用

- [x] 实现历史数据适配层，复用现有 UI 的点云、mask、OBJ/坐标轴和筛查显示。
- [x] 加入 RGB/首行查看、记录导航及可配置数据地址，兼容原查看器和 HTML 导出。
- [x] CPU 转换有界执行，缓存可重建，回看不占 GPU worker。

### 阶段 5：realman_pi 集成

- [x] 补充请求元数据和结果 history 字段透传。
- [x] 增加入口和本次结果链接、同源代理、配置及启动接线；不复制历史业务实现。
- [x] 完成实际浏览器端“采集 → 注册 → 列表 → 详情 → 重启后回看”。

### 阶段 6：旧数据与交付检查

- [x] dry-run 并导入 `multi_inbox08` 的首帧实例，校验输入匹配和缺失提示。
- [x] 交付部署、备份/恢复、模型快照和容量说明；更新项目 readme。

关键验证要求：

| 范围 | 验收依据 |
| --- | --- |
| 时间主键 | 同纳秒并发、墙钟回拨、进程重启后不覆盖；JSON/URL/浏览器保留完整纳秒 ID；时间筛选边界一致 |
| 请求身份 | 同请求同摘要去重、同 ID 不同输入拒绝；同 observation 的新请求独立保存 |
| 预览 | 样本首行恰好 409 × 100，像素与原图一致；RGB 无红蓝互换；首行候选/分数与返回 best pose 对应，包括同分情况 |
| 输入可追溯 | 原始 float32 深度无损；RGB/mask/K 与请求一致；旧档案不依赖已更换的模型挂载 |
| 点云几何 | 已知 RGB-D/K 可核对反投影、stride、颜色与米制单位；原始 OBJ 变换无二次居中 |
| 生命周期 | 成功、异常、稀疏深度、超时、断线、崩溃恢复均有明确状态；不会读取另一请求的 debug 图 |
| 容器持久化 | 删除并重建容器后，原历史页面、RGB/score、点云和模型仍可用 |
| 页面 | 时间分页筛选、双图预览、详情刷新/深链/后退、点云交互与缺失状态可用；参考查看器现有测试仍通过 |
| 接入边界 | `realman_pi` 仅有客户端、链接、代理和部署接线；FoundationPose 独立地址可完整浏览 |
| 性能 | 用真实冻结 RGB-D 对比 debug=0 基线与 debug=2，报告 GPU/可视化/归档耗时、峰值内存和单条落盘体积；同时回看不阻塞新推理受理，不造成无界排队 |

实现时为主键/恢复、图像身份、几何和服务生命周期增加有行为意义的自动化验证；页面做浏览器端到端检查。运行 `tests/test_multi_instance_viewer.py` 等受影响的既有检查，确认原多实例查看器没有回归。


## 11. 实施验收记录（2026-11-17）

- FoundationPose Python 检查 32 项通过，RealMan 客户端/代理相关检查 54 项通过，原 6D 页面桌面/移动浏览器回归 12 项通过；历史浏览器检查通过真实 HTTP 与 Web 代理。
- `multi_inbox08` 的 19 个首帧实例已导入。独立 GPU 测试完成两次注册，并验证删除/重建容器后 21 条记录与原点云仍可读取。
- 线上已更新 `pose6d_server`、`realman_web_control` 和 `pose6d_bridge`。持久化服务以 UID/GID 1000 运行，使用只读根目录、专用可写挂载和 `cap_drop: ALL`。
- 真实右腕相机通过 `/6d_pose.html` 完成 848×530 采集、分割、实例选择、注册、历史链接与详情；记录 `1794885176694644098`，归档 RGB 与页面冻结 RGB 逐像素一致。浏览器实际发送的命令只有 capture/segment/preview/infer。
- 同一 640×480 输入：原 `debug=0` 服务约 2.27 秒；新 `debug=2` 首次约 18.18 秒、热运行约 16.59 秒，其中最终归档约 315 ms，归档体积约 55.3 MB。两种 debug 下 pose/score 完全一致，最高分首行裁剪逐像素一致。
- 真实 848×530 注册耗时约 16.13 秒，归档约 64.4 MB，Torch 峰值 allocated 约 6.07 GB；同时读历史列表约 33 ms。这些数据来自当前 RTX 4080 和实际输入，不作为所有场景的性能承诺。
- 原始模型几何完整快照，查看器使用实例配色；首版不保留纹理外观。每次历史详情显示该次注册实例，跨注册多实例聚合留待后续。

- 线上算法服务收到 SIGTERM 后可正常退出并完成重启；重启后 22 条历史及真实相机记录点云仍可通过 Web 代理读取。
