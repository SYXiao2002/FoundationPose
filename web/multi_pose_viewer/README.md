# 多实例首帧 Pose 可视化

在 FoundationPose 根目录启动：

```bash
.venv/bin/python visualize_multi_instance_web.py --no-browser
```

浏览器打开终端输出的地址，默认是 `http://127.0.0.1:8765/`。端口占用时自动尝试后续端口。VS Code Remote 可转发对应端口。

默认读取 `results/multi_inbox03/instance*/ob_in_cam/` 中各实例的第一份 pose，使用 `demo_data/eu-plug/multi_scene_eu-plug/multi_inbox03_eu-plug/raw/` 的同名 RGB-D 和 `cam_K.txt`，以及 `demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj`。

- 左侧复选框选择一个或多个 instance；支持全选、清空、反选。
- Mask 图标出真实 instance ID，支持点击、拖框选择，以及按住 Shift 追加。点击“放大”可以在大图上选择。
- 顶部“坐标系”控制所选实例的坐标轴，红 X、绿 Y、蓝 Z；坐标轴长度可调。
- 顶部“OBJ”控制模型显示，左侧每行的 OBJ 开关进一步控制单个实例。模型不透明度可调至 0–100%。OBJ 使用实例配色，方便区分。
- 默认只显示所有物体附近的点云；取消“物体附近点云”可查看完整场景。
- 左键旋转、右键平移、滚轮缩放；支持正视、侧视和聚焦所选。切换实例或 OBJ 时保留当前视角。
- 底部显示所选 pose 原点的相机坐标（米）；悬停实例列表行可查看更精确的平移和原始 register score。

导出可直接用浏览器打开的单文件 HTML：

```bash
.venv/bin/python visualize_multi_instance_web.py \
  --output results/multi_inbox03/pose_viewer.html
```

导出文件内嵌点云、模型、mask、位姿和 Plotly，无需服务、网络或额外资源，可以复制到其他电脑查看。需要支持 WebGL 的浏览器。

其他场景通过参数指定，OBJ 必须与估计 pose 时使用的模型一致：

```bash
.venv/bin/python visualize_multi_instance_web.py \
  --debug-dir results/another_scene \
  --scene-dir demo_data/another_scene/raw \
  --mesh-file demo_data/object/mesh.obj \
  --point-stride 2 \
  --port 8765 --no-browser
```

点云来自原始首帧 RGB-D，而非各 instance 的局部 `scene_raw.ply`。深度 PNG 按毫米转换为米，反投影使用原始像素坐标和内参；`ob_in_cam` 直接将原始 OBJ 顶点变换到相机坐标，不进行额外居中或缩放。各 instance 的首帧必须相同，保存的注册 RGB 必须与指定 RGB 场景一致，否则启动时会报错。

默认每隔 2 个像素采样；`--point-stride 1` 可使用完整分辨率。仅第一帧会被读取，没有时间轴或后续帧功能。

数据检查：

```bash
.venv/bin/python -m unittest discover -s tests -p test_multi_instance_viewer.py -v
```
