# 6D 位姿历史

FoundationPose Docker 同时提供原有 MessagePack WebSocket（8092）与历史 HTTP（8095）。线上 `register()` 固定为 `debug=3`，保留 `vis_score.png` 并跳过 refiner montage；每次推理独立保存，不受浏览器退出、断线或请求超时影响。

独立运行时 `debug=2` 仍生成 `vis_refiner.png` 和 `vis_score.png`；`debug=3` 是轻量评分诊断模式，只生成 `vis_score.png`。两个级别使用相同的候选姿态、细化迭代和评分计算。已有历史中的 refiner 图片仍可读取。

Pose6D server 支持重复传入 `--model ID=OBJ_PATH` 注册多个 OBJ；每条请求用 `object_id` 和对应 OBJ SHA-256 选择模型，`hello.models` 会公布所有已注册模型身份。GPU 上只保留当前模型，切换物体时释放旧 FoundationPose 实例并加载新 OBJ；历史会分别快照并绑定各 OBJ 几何。单模型 `--mesh` 仍兼容旧启动方式。RealMan 页面用独立的“采集分割对象”和“解算 OBJ”下拉框：前者固定为欧标插头，后者从 bridge 配置读取可用 OBJ。一帧采集和分割结果可供多个 OBJ 反复解算；每次解算独立传递模型 ID，各次历史共享 observation/实例身份并分别记录所用模型。

例如新增的 `demo_data/eu-plug-1124/mesh/eu-plug-1124mtl.obj` 已登记为 `eu_plug_1124`，可以计算和审阅 6D 位姿。该目录目前没有该物体的吸盘接触区域/法线 JSON，因此 UI 会明确禁用吸盘运动；在补齐并验证此物体自己的接触标注前，不应复用欧规插头的接触点。

历史页面：`http://<robot-host>:8765/6d_pose_history/`。页面由 FoundationPose 提供，RealMan Web 只代理；也可在算法宿主机打开 `http://127.0.0.1:8095/6d_pose_history/`。每条记录展示原始 RGB 和最高分评分首行，详情复用多实例查看器，包含 RGB-D 点云、OBJ、坐标系、mask 和当次筛查结果。

总览卡片和每条记录的详情都可将人工判断设为“待判断 / 正确 / 错误”。每次选择会立即保存到历史根目录的 `reviews/<history_id>.json`，重启服务后仍保留；清除判断也会写入文件，留下显式的“待判断”状态。

## 构建与启动

从相邻 `realman_pi` 仓库执行（默认 UID/GID 1000；目录由该用户创建）：

```bash
mkdir -p ../FoundationPose/data/pose_history
docker compose --project-directory . -p realman_pi \
  -f config/docker/compose.pose6d.yaml --profile build-base build pose6d_runtime_base
docker compose --project-directory . -p realman_pi \
  -f config/docker/compose.pose6d.yaml build pose6d_server
docker compose --project-directory . -p realman_pi \
  -f config/docker/compose.pose6d.yaml up -d pose6d_server
```

`pose6d_runtime_base` holds the fixed CUDA/Python dependencies and compiled nvdiffrast extension. Build it when the base image or requirements change; source-only updates rebuild only the small server image. From `realman_pi`, `./scripts/build_pose6d.sh` runs these two builds in order and omits changing provenance attestations so the tagged base image keeps a stable digest and downstream CMake layers remain cached.

Dockerfile 和算法依赖位于 FoundationPose 的 `docker/pose6d-server.Dockerfile` 与 `docker/pose6d-requirements.txt`。`POSE6D_REFERENCE_REPO` 指定本仓库构建上下文；GPU 基础镜像、PyTorch3D 和 nvdiffrast 参数沿用原部署。已有本地 BundleSDF 镜像时可使用 `--build-arg POSE6D_BASE_IMAGE=nvcr.io/nvidian/bundlesdf:latest --build-arg POSE6D_USE_PREINSTALLED_PYTORCH3D=1`，nvdiffrast 使用 Compose 内固定提交。

RealMan Web 与 bridge 修改后需要重建 ROS 镜像并重启对应两个服务；配置 `config/ros/realman_web_control.yaml` 的 `pose6d_history.url` 指向固定算法 HTTP origin。它们继续通过原有 8092 调用推理，只增加拍摄/实例元数据和历史结果链接。

Compose 默认将 `FoundationPose/data/pose_history` 挂载至容器 `/data/pose_history`。可设置 `POSE6D_HISTORY_DIR` 选择宿主机目录，`POSE6D_HISTORY_HTTP_PORT` 修改宿主机 HTTP 端口，`POSE6D_HISTORY_BIND_HOST` 默认 `127.0.0.1`。修改宿主机端口后同步修改代理目标。容器以 `POSE6D_UID:POSE6D_GID`（默认 `1000:1000`）运行，历史宿主机目录必须属于该用户；首次启动前创建该目录。容器内部 HTTP 端口仍为 8095；直接运行 Python 服务时可通过同名环境变量或 `--http-port` 更改。

`GET /healthz` 返回历史记录数、归档体积、磁盘剩余空间、存储可用状态及 GPU busy 状态。历史目录启动时不可写会导致服务启动失败；运行中归档失败会单独返回 `archive_status=failed` 并停止接受新归档。修复磁盘问题后重启服务，启动恢复会保留已写入的可用产物。

## 数据与时间约定

- 一次实际注册对应一条记录；相同 observation 的重复解算和不同实例各自保留。相同 request ID 的相同输入返回原记录，不重新计算；不同输入使用相同 ID 被拒绝。
- UTC 纳秒时间是 SQLite 主键，持久化高水位保证同纳秒、时钟回拨和重启不覆盖。人工判断单独写入 `reviews/<history_id>.json`，不依赖 SQLite 数据库字段。JSON 与 URL 中时间均为字符串，页面按上海时间展示。
- 原始拍摄时间单独保存，并附时钟来源；ROS timestamp 不自动解释成系统日历时间。
- `RGB.png`、`depth_m.npy`（float32、米）、K、实际 mask、原始 OBJ 坐标下的 pose、模型几何快照和 debug=3 产物都归档。每个记录 manifest 含摘要、参数、筛查阈值、代码/权重版本和文件可用性。
- 模型几何按摘要去重保存在 `models/`，不依赖当前模型挂载；查看器用原始 RGB-D 反投影，不使用算法滤波后的 debug 深度。展示使用实例配色，模型快照不保留纹理外观。
- 评分图首行完整保留四列和分数文字；当前格式为 100 像素高，不含 5 像素分隔。分数是原始网络输出，不是概率。
- 列表是 cursor 分页，`from` 包含、`to` 不包含。后台按记录加载点云，只有两个 CPU 展示线程和四个任务名额；不占 GPU 推理 worker。

完整 API 约定见 [实施计划](plans/2026-11-17-6d-pose-history.md)。WS 结果增加 `history_id`、`history_path`、`archive_status`；原有 pose/score/screening 字段保持兼容。

6D 采集页将 RGB、Depth、全部实例 ID 总览及所选实例预览放在点云上方。总览中的 `#编号` 与实例选择列表一致，选择实例或解算后仍保留总览。“查看本次历史”在新标签页打开，保留采集页的当前观测与复核状态。历史详情默认显示 OBJ，仍可通过查看器开关隐藏。

## 导入已有结果

先检查，再加 `--execute` 导入。导入复制输入和模型快照，重复执行按源记录身份去重。缺失或不匹配的输入会在报告中列出，不用其他图片代替。

```bash
.venv/bin/python -m pose_history.import_legacy \
  --root data/pose_history \
  --debug-dir results/multi_inbox08 \
  --scene-dir demo_data/eu-plug/multi_scene_eu-plug/multi_inbox08_eu-plug/raw \
  --mesh-file demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj
# 检查报告后，重复上面的命令并添加 --execute
```

每个实例导入为一条历史，保留同一 observation 关联。导入主键使用导入时间，原帧时间保存在 `capture_timestamp_ns`。当前浏览详情按单次注册显示一个实例。

只浏览已有历史，无需 GPU：

```bash
.venv/bin/python -m pose_history.http --root data/pose_history --port 8095
```

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_pose_history.py tests/test_pose_history_service.py \
  tests/test_pose_service_protocol.py tests/test_multi_instance_viewer.py -q
npm install --prefix tests/browser
node tests/browser/check_history.cjs http://127.0.0.1:8095
node tests/browser/check_history_cache.cjs http://127.0.0.1:8765
```

浏览器检查需要 Chrome（默认 `/usr/bin/google-chrome`，可设 `PLAYWRIGHT_CHROME_PATH`）和至少一条完成记录；截图保存至 `/tmp/pose-history-{list,detail,mobile}.png`。

## 备份与容量

首版不会自动删除历史，debug 评分/细化长图可能占用较大空间。通过 `/healthz` 查看记录体积和磁盘余量，并预留模型及派生缓存的空间。停止算法服务后，整体备份历史挂载目录（SQLite、WAL/SHM、`records/`、`reviews/`、`models/`、`staging/`）；恢复整个目录即可，不需要恢复原始模型绝对路径。在线备份应使用 SQLite backup API 并协调文件快照，不能仅复制正在写入的数据库主文件。

`records/.../cache/` 可在停服务后删除，后续按需重建。不要单独删除记录文件或 `models/`，否则摘要验证会报告档案损坏。`staging/` 的残留会在独占服务启动时恢复为中断记录；孤立目录只记录告警，不自动丢弃。


## 当前验证结果（2026-11-17）

已在实际 Docker、只读根目录、UID/GID 1000 和 `cap_drop: ALL` 下完成右腕相机端到端注册与历史回看。已导入 `multi_inbox08` 的 19 条首帧实例。32 项 FoundationPose Python 检查、54 项 RealMan 客户端/代理检查与原页面的 12 项浏览器回归通过；历史列表/详情另有桌面及移动浏览器检查。

图像布局、实例编号、新标签页和默认 OBJ 调整后，8 项历史服务测试及扩展后的 12 项采集页浏览器回归通过。实际右腕相机的 16 个实例总览、选择后保留总览、解算后新标签页跳转及 OBJ 默认渲染均已验证（历史记录 `1794886827375000649`）。

历史性能基线记录的是完整 debug=2：同一 640×480 输入的旧 debug=0 注册约 2.27 秒，debug=2 热运行约 16.59 秒（最终归档约 315 ms、55.3 MB）；pose 和 score 完全一致。真实 848×530 帧注册约 16.13 秒、64.4 MB，推理中读取历史列表约 33 ms。当前服务改用 debug=3，不再构建或归档 refiner montage。

## 评分候选预览

`6d_pose.html` 收到解算结果后，用该结果的 `history_id` 加载与历史页相同的 `preview/score`，显示最高分候选的 ID、分数和 RGB / Depth 对比。重新采集、修改 Mask 或切换实例后清除旧候选图；归档或图片读取失败时只显示候选图不可用提示。

历史详情页保留左侧现有最高分候选展示，在三维视图最右侧新增独立滚动的“vis_score · 前 30 候选”栏，小屏幕时排到三维视图下方。新接口 `GET /6d_pose_history/api/records/{id}/preview/score-top30` 从已归档 `debug/vis_score.png` 读取前 30 个完整候选行；每行 100 px，行距 5 px，最多输出 3145 px 高。原图已经按评分降序排列，图内保留候选 ID、原始评分和四格对比。候选不足 30 个时按实际数量显示。

该接口只读取并校验已有归档，不修改 manifest 或补写旧文件，也不重新运行 GPU 解算，因此已有历史同样支持。`preview/score` 仍输出原来完全相同的最高分一行。

历史列表与详情的本地脚本、样式地址包含文件内容摘要，部署后普通刷新即可加载匹配版本；资源响应使用 `Cache-Control: no-cache`。这避免已访问过历史页的浏览器沿用旧两栏 CSS/脚本，导致候选栏掉到左下角且停在“等待评分候选图”。缓存升级浏览器检查先缓存旧版页面资源，再打开新版页面，覆盖 1440、1024 和 412 px 宽度。
