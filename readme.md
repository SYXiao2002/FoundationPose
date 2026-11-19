# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

\
**🥇 No. 1 on the world-wide [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/) (as of 2024/03) for model-based novel object pose estimation.**
<img src="assets/bop.jpg" width="80%">

## 6D pose history

The containerized single-frame service saves debug=3 outputs and timestamp-keyed RGB-D/pose history. See [deployment and usage](docs/6d_pose_history.md).

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe



# Bibtex
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

1) [Optional] Download our large-scale training data: ["FoundationPose Dataset"](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

1) [Optional] Download our preprocessed reference views [here](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing) in order to run model-free few-shot version.

# Env setup option 1: docker (recommended)
  ```
  cd docker/
  docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
  bash docker/run_container.sh
  ```


If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).
In short, do the following:
```
docker pull shingarey/foundationpose_custom_cuda121:latest
```
Then modify the bash script to use this image instead of `foundationpose:latest`.


# Env setup option 2: conda (local)

1) **Create the environment** (C++ build deps + Python; all on `conda-forge`):

```bash
conda env create -f environment.yml
conda activate foundationpose
```

2) **Install PyTorch** with a CUDA build that matches your machine. The [PyTorch “Get Started”](https://pytorch.org/get-started/locally/) page lists the right `--index-url` (for example `cu124` works on most current NVIDIA drivers). Example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3) **Install PyTorch3D and NVDiffRast** (compile from source; needs the CUDA toolkit for `nvcc`. Point `CUDA_HOME` at your install, e.g. `/usr/local/cuda-12.8` or `/usr/local/cuda`):

```bash
export CUDA_HOME=/usr/local/cuda   # or e.g. /usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
```

`--no-build-isolation` is required so the build can see the `torch` you already installed.

4) **Install remaining Python dependencies** and **build the `mycpp` extension** (BundleSDF’s `mycuda` step is optional; skip it unless you need the model-free / NeRF path):

```bash
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

5) **Optional — Kaolin** (only for the model-free setup; version must match your PyTorch/CUDA—see the [Kaolin install docs](https://kaolin.readthedocs.io/en/latest/notes/installation.html)).


# Env setup option 3: uv (local)

`uv` is a fast Python package/environment manager (a `pip` + `venv` replacement), but it cannot provide native C++ libraries. So this option is a **hybrid**: use the system package manager (`apt`) for the compiler / Eigen / Boost, and `uv` for everything Python.

1) **Install native C++ build dependencies via `apt`** (replaces conda's `cxx-compiler`, `eigen`, `boost-cpp`):

```bash
sudo apt update
sudo apt install -y build-essential libeigen3-dev \
    libboost-system-dev libboost-program-options-dev
```

2) **Create the environment and install build tooling with `uv`** (`cmake`, `ninja`, `pybind11` come from PyPI wheels, replacing the conda equivalents):

```bash
# Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install cmake ninja pybind11 setuptools wheel
```

> **Note:** unlike a conda env, a fresh `uv venv` does **not** ship `setuptools`. It must be installed explicitly, otherwise the `--no-build-isolation` source builds in step 4 fail with `ModuleNotFoundError: No module named 'setuptools'`.

3) **Install PyTorch** with a CUDA build that matches your machine (see the [PyTorch “Get Started”](https://pytorch.org/get-started/locally/) page for the right `--index-url`):

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4) **Install PyTorch3D and NVDiffRast** (compile from source; needs a full **CUDA Toolkit** for `nvcc`). `--no-build-isolation` is required so the build can see the `torch` you already installed.

> **Prerequisite:** a system CUDA Toolkit whose version **matches the PyTorch CUDA build** (here `cu124` → Toolkit 12.4). Note `nvidia-smi`'s "CUDA Version" is only the *maximum* runtime your driver supports, not an installed toolkit. `uv` cannot provide `nvcc`, so install it system-wide, e.g. via NVIDIA's apt repo:
>
> ```bash
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
> sudo dpkg -i cuda-keyring_1.1-1_all.deb
> sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-4
> ```

```bash
export CUDA_HOME=/usr/local/cuda-12.4   # match your torch CUDA build
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.9"       # optional: your GPU arch, speeds up compilation
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
```

5) **Install remaining Python dependencies** and **build the `mycpp` extension**:

```bash
uv pip install -r requirements.txt
bash build_all_uv.sh
```

`build_all_uv.sh` mirrors `build_all_conda.sh`, but locates Python via `VIRTUAL_ENV` and injects pybind11's CMake dir into `CMAKE_PREFIX_PATH` (Eigen3/Boost are found from the system `apt` install).

6) **Verify the environment** — this checks PyTorch/CUDA, all Python deps, and the compiled `mycpp` extension:

```bash
python check_env.py
```

7) **Optional — Kaolin** (only for the model-free setup; version must match your PyTorch/CUDA—see the [Kaolin install docs](https://kaolin.readthedocs.io/en/latest/notes/installation.html)).


# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```

## Record RGB-D data with MobileSAM masks

`record_dataset.py` records aligned Orbbec colour/depth frames in the directory format expected by `run_demo.py`. It prompts MobileSAM (the same MobileSAM package and checkpoint used by the offline grasp tools) with an object box selected on the first frame, then saves a binary mask for each recorded RGB-D pair. The Orbbec/MobileSAM environment is separate from the FoundationPose inference environment; pass `--sam-model` if the checkpoint is stored elsewhere.

```bash
/home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/.venv/bin/python record_dataset.py \
  --object eu-plug --data-dir '/home/yixiaoshen/Repo/FoundationPose/demo_data' --serial 'CV2856D0002E' --fps 30 --record-fps 30 --duration 300

source .venv/bin/activate
python run_demo.py \
  --mesh_file "$PWD/demo_data/eu-plug-mesh/TexturedMeshSimplified.obj" \
  --test_scene_dir "$PWD/demo_data/<timestamp>_eu-plug" \
  --debug 2 --debug_dir "$PWD/debug/eu-plug"

python visualize_pose_3d.py \
  --debug-dir debug/eu-plug-12 \
  --mesh-file demo_data/eu-plug-mesh/TexturedMeshSimplified.obj \
  --crop-padding 0.05
```

The preview starts paused. Select the target in the first frame, press `SPACE`/`r` to start or pause, and press `b` to select a new box. Press `q` to finish. For headless recording, supply `--bbox x1 y1 x2 y2 --no-preview`; coordinates are in the colour image. Saved depth PNGs are `uint16` millimetres, and `cam_K.txt` matches the saved aligned colour resolution.

### Orbbec 深度质量设置

`record_dataset.py` 直接调用官方 Python SDK（安装包 `pyorbbecsdk2`，导入名 `pyorbbecsdk`），不经过 ROS。当前相机环境安装版本为 2.1.2。处理顺序为：原生深度 → SDK 滤波 → 对齐至 RGB → 距离范围筛选 → 保存为 `uint16` 毫米 PNG；`cam_K.txt` 始终对应保存的 RGB 分辨率。

默认 RGB 和原生深度流均为 `848×530`，与 realman_pi 右腕 Gemini 305 的当前档位一致；保存的 RGB、对齐深度和 mask 也均为 `848×530`。可通过 `--width`、`--height` 和 `--depth-width`、`--depth-height` 分别指定其他相机支持的档位。

针对固定相机拍静止料箱/插头、重点改善 FoundationPose 注册首帧的场景，默认 `--depth-filter first-frame`。先丢弃 60 对预热帧；开始录制后，采集 30 张经过保边空间滤波并对齐至 RGB 的深度，对每个像素的有效测量取中位数，抑制随机噪声和偶发异常值，再保存为第一帧 depth。首帧 RGB 和 mask 来自采集窗口的最后一帧，因此整个采集窗口内相机和物体必须保持静止。后续帧使用相机未加软件滤波的对齐深度，继续供跟踪使用。

每个像素至少需要在采集窗口内 80% 的帧中有有效深度，而且窗口最后一帧中也必须有效，否则输出 0。关闭补洞，避免将缺测直接填成附近背景。选框或按开始/暂停时会清空尚未完成的首帧采样，确保首帧在开始录制后重新积累。录制示例：

```bash
INBOX_ID=14
/home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/.venv/bin/python record_dataset.py \
  --object eu-plug --data-dir '/home/yixiaoshen/Repo/FoundationPose/demo_data/eu-plug/multi_scene_eu-plug/multi_inbox${INBOX_ID}_eu-plug/raw' --serial 'CV2856D0002E' --fps 30 --record-fps 30 --duration 300
```

如果只想保存这一张高质量初始化图，追加 `--max-frames 1`。首帧融合详情和 SAM 区域的有效深度比例会记录在 `capture_meta.json` 的 `first_frame_depth` 中。中位数融合主要降低随机噪声，无法保证绝对测量精度，也无法恢复反光、遮挡或过近距离造成的真实缺测。

还可使用 `--depth-filter spatial` 对整段序列做保边空间滤波，或 `--depth-filter static` 对整段序列做空间滤波和 SDK 时间滤波；`static` 在有运动时可能产生拖影。`--depth-filter none` 关闭新增的软件滤波。此前实机 60 帧的相邻帧差改善（均值 2.28→0.98 mm，95 分位 7→2 mm）对应整段 `static` 时间滤波，不是首帧中位数融合的精度验证。

| 参数 | 用途 |
| --- | --- |
| `--first-frame-samples 30` | 默认模式下用于首帧中位数融合的深度帧数，30 FPS 时名义采样时间约 1 秒 |
| `--first-frame-min-valid 0.8` | 首帧融合中，每个像素所需的有效测量比例 |
| `--spatial-alpha 0.5` | 当前像素权重，范围 0.1–1；减小会增强空间平滑 |
| `--temporal-weight 0.4` | `static` 模式的当前帧权重，范围 0.1–1；减小会增强时间平滑并增加拖影风险 |
| `--depth-min-mm 100 --depth-max-mm 1500` | 示例距离范围；范围外置 0，按实际工作距离调整，不会填补缺测 |
| `--save-unfiltered-depth` | 同时在 `depth_raw/` 保存同帧、同 RGB 坐标系的未滤波深度，仍为 `uint16` 毫米，可与 `depth/` 对比 |
| `--depth-width 1280 --depth-height 800 --depth-fps 30` | 指定原生深度档位；输出仍对齐到所选 RGB 分辨率。仅在相机支持时可用，会增加传输和处理开销 |
| `--camera-preset 'High Accuracy'` | 可选的相机预设，正常退出或 Ctrl+C 时恢复之前的预设；效果需按场景对比 |

深度帧率默认匹配实际选中的 RGB 帧率。指定不支持的 RGB 分辨率或深度档位会报错，不会静默更换分辨率。可查询所选相机的档位和预设，不启动录制：

```bash
/home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/.venv/bin/python record_dataset.py \
  --serial CV2856D0002E --list-profiles
```

相机 `CV2856D0002E` 为 Gemini 305，已查询到 `1280×800@30` 深度档位，以及 `Default`、`High Accuracy`、`Close Range High Accuracy` 等预设。多台主动深度相机同时照射同一料箱时，也应检查是否存在相互干扰。实际使用的 SDK 版本、流档位、滤波参数、距离范围和所请求预设都会记录在 `capture_meta.json` 中；已有数据不会被自动修改。

## 从 X-AnyLabeling 标注生成多实例数据

以下命令均在 FoundationPose 仓库根目录执行，使用 `.venv` 中的 Python。将首帧 RGB 复制出来，在 X-AnyLabeling 中逐个分割目标，保存 JSON 和对应 PNG。`raw/` 应包含以下文件，其中 `rgb/`、`depth/` 使用相同的帧名，深度 PNG 为 `uint16` 毫米单位：

```text
demo_data/eu-plug/multi_scene_eu-plug/multi_inbox04_eu-plug/
└── raw/
    ├── inbox04.json
    ├── inbox04.png
    ├── cam_K.txt
    ├── capture_meta.json          # 可选，采集元数据
    ├── rgb/<帧时间戳>.png
    └── depth/<帧时间戳>.png
```

运行 [prepare_instance_datasets.py](prepare_instance_datasets.py)：

```bash
INBOX_ID=13
.venv/bin/python prepare_instance_datasets.py \
  --annotation demo_data/eu-plug/multi_scene_eu-plug/multi_inbox${INBOX_ID}_eu-plug/raw/inbox${INBOX_ID}.json
```

脚本检查标注 PNG 与按文件名排序后的首帧 RGB 完全一致，筛选 `eu-plug` 标签，并按 JSON `shapes` 中的顺序从 1 编号。每个匹配的 shape 对应一个实例；`polygon` 和作为分割轮廓导出的 `linestrip` 均按闭合多边形填充为 0/255 二值 mask。默认在 `raw/` 的上一级生成：

```text
multi_inbox04_eu-plug/
├── raw/
├── single_inbox1_eu-plug/
│   ├── rgb/                      # 完整 RGB 序列
│   ├── depth/                    # 完整深度序列
│   ├── masks/<首帧时间戳>.png     # 仅该实例的首帧 mask
│   ├── cam_K.txt
│   └── capture_meta.json         # 含实例 ID、原始 shape 下标和 mask 信息
├── single_inbox2_eu-plug/
└── ...
```

`run_demo.py` 使用首帧 mask 完成姿态初始化，后续帧由跟踪器处理，因此只需要首帧 mask。当前 `inbox03` 已生成 17 个实例，每个有 26 对 RGB/depth；`inbox04` 已生成 14 个实例，每个有 46 对 RGB/depth。

处理其他场景时替换 `--annotation` 路径即可；`--label` 可选择其他类别，`--output-dir` 可指定输出父目录。已有同名实例目录时脚本会拒绝覆盖；当前 `inbox03`、`inbox04` 已完成转换，无需重复生成。

## 批量运行多实例 demo

[run_demo_batch.py](run_demo_batch.py) 自动发现实例目录，按编号依次调用 `run_demo.py`，每次运行一个实例。所有实例共用同一 mesh，使用当前 Python 环境执行，并关闭弹窗预览。

先预览待执行命令，`--dry-run` 不运行推理，也不创建输出：

```bash
.venv/bin/python run_demo_batch.py \
  --scene-root demo_data/eu-plug/multi_scene_eu-plug/multi_inbox04_eu-plug \
  --mesh-file demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj \
  --output-dir results/multi_inbox04 \
  --dry-run
```

批量运行 `inbox04` 的 14 个实例：

```bash
INBOX_ID=13
.venv/bin/python run_demo_batch.py \
  --scene-root demo_data/eu-plug/multi_scene_eu-plug/multi_inbox${INBOX_ID}_eu-plug \
  --mesh-file demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj \
  --output-dir results/multi_inbox${INBOX_ID}
```

默认 `--debug 2`，保存逐帧姿态、可视化 PNG 和 H.264 MP4。视频使用 `yuv420p` 和 `faststart`，可以在 VS Code 中直接预览。需要系统安装带 `libx264` 编码器的 FFmpeg（Ubuntu：`sudo apt install ffmpeg`）。输出结构如下，`instance<ID>` 与输入 `single_inbox<ID>_eu-plug` 对应，也兼容现有多实例查看器：

```text
results/multi_inbox04/
├── instance1/
│   ├── ob_in_cam/<帧时间戳>.txt   # 4×4 位姿矩阵
│   ├── track_vis/<帧时间戳>.png
│   ├── register_vis.png
│   ├── register_score.json
│   └── pose_video.mp4
├── instance2/
├── ...
├── logs/
│   ├── instance1.log
│   └── ...
└── batch_summary.json            # 状态、命令、耗时及日志路径
```

可在上述命令末尾追加以下参数：

| 参数 | 用途 |
| --- | --- |
| `--instances 1 3 5` | 仅处理指定实例，按编号排序运行 |
| `--pattern 'single_inbox*_eu-plug'` | 指定实例目录匹配规则，默认即此规则 |
| `--debug 0` | 仅保存姿态及推理过程输出，不生成视频 |
| `--est-refine-iter 5 --track-refine-iter 2` | 设置首帧姿态估计和后续跟踪的迭代次数 |
| `--video-fps 20` | 手动指定视频帧率，默认根据输入时间戳推算 |
| `--overwrite` | 强制重跑已有完整结果的实例 |

默认检查每帧姿态文件；`--debug 2` 及以上还检查每帧可视化 PNG 和 MP4 是否存在且非空，满足条件的实例会跳过。重复执行相同命令可以继续处理尚未完成的实例。脚本按文件完整性判断是否跳过；更换 mesh 或推理参数后，请使用新输出目录或追加 `--overwrite`。实际重跑的实例会清空其 `instance<ID>/` 目录，日志也会重写。

单个实例失败后会继续处理后续实例，具体错误可查看 `logs/instance<ID>.log`。每个实例结束后更新 `batch_summary.json`；只要有失败项，批处理最终退出码为 1。将示例中的场景路径改为 `multi_inbox03_eu-plug`、结果路径改为 `results/multi_inbox03`，即可处理 `inbox03`。


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Run on public datasets (LINEMOD, YCB-Video)

For this you first need to download LINEMOD dataset and YCB-Video dataset.

To run model-based version on these two datasets respectively, set the paths based on where you download. The results will be saved to `debug` folder
```
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```

To run model-free few-shot version. You first need to train Neural Object Field. `ref_view_dir` is based on where you download in the above "Data prepare" section. Set the `dataset` flag to your interested dataset.
```
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```

# Troubleshooting


- For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).

- For setting up on Windows, refer to [this](https://github.com/NVlabs/FoundationPose/issues/148).

- If you are getting unreasonable results, check [this](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043) and [this](https://github.com/030422Lee/FoundationPose_manual)

# Training data download
Our training data include scenes using 3D assets from GSO and Objaverse, rendered with high quality photo-realism and large domain randomization. Each data point includes **RGB, depth, object pose, camera pose, instance segmentation, 2D bounding box**. [[Google Drive]](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing).

<img src="assets/train_data_vis.png" width="80%">

- To parse the camera params including extrinsics and intrinsics
  ```
  glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)
  W, H = camera_params["renderProductResolution"]
  with open(f'{base_dir}/camera_params/camera_params_000000.json','r') as ff:
    camera_params = json.load(ff)
  world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
  cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
  world_in_cam = np.linalg.inv(cam_in_world)
  focal_length = camera_params["cameraFocalLength"]
  horiz_aperture = camera_params["cameraAperture"][0]
  vert_aperture = H / W * horiz_aperture
  focal_y = H * focal_length / vert_aperture
  focal_x = W * focal_length / horiz_aperture
  center_y = H * 0.5
  center_x = W * 0.5

  fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
  K = np.eye(3)
  K[0,0] = fx
  K[1,1] = fy
  K[0,2] = cx
  K[1,2] = cy
  ```



# Notes
Due to the legal restrictions of Stable-Diffusion that is trained on LAION dataset, we are not able to release the diffusion-based texture augmented data, nor the pretrained weights using it. We thus release the version without training on diffusion-augmented data. Slight performance degradation is expected.

# Acknowledgement

We would like to thank Jeff Smith for helping with the code release; NVIDIA Isaac Sim and Omniverse team for the support on synthetic data generation; Tianshi Cao for the valuable discussions. Finally, we are also grateful for the positive feebacks and constructive suggestions brought up by reviewers and AC at CVPR.

<img src="assets/cvpr_review.png" width="100%">


# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.


# Contact
For questions, please contact [Bowen Wen](https://wenbowen123.github.io/).
