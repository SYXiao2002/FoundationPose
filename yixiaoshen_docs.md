/home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/.venv/bin/python record_dataset.py \
  --object eu-plug --data-dir '/home/yixiaoshen/Repo/FoundationPose/demo_data' --serial 'CV2856D0002E' --fps 30 --record-fps 30 --duration 300


cd /home/yixiaoshen/Repo/FoundationPose
source .venv/bin/activate
python run_demo.py \
  --mesh_file "$PWD/demo_data/eu-plug-mesh/TexturedMeshSimplified.obj" \
  --test_scene_dir "$PWD/demo_data/2026-11-04-09-12-56_eu-plug" \
  --debug 2 --debug_dir "$PWD/debug/eu-plug-12"

cd /home/yixiaoshen/Repo/FoundationPose
source .venv/bin/activate

python visualize_pose_3d.py \
  --debug-dir results/multi_inbox03/instance8 \
  --mesh-file demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj \
  --crop-padding 0.05