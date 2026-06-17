# 用于测试双臂piper的连接是否成功
 
python src/test_bi_piper_hardware.py

conda activate lerobot

python -m lerobot.robots.bi_piper.openpi_client \
  --host "192.168.10.2" \
  --port 8000 \
  --task "Agilex_Cobot_Magic_pick_up_bottle_2" \
  --frequency 30 \
  --action_type absolute \
  --robot.type bi_piper \
  --robot.can_left can0 \
  --robot.can_right can1 \
  --robot.velocity 10 \
  --robot.init_type joint \
  --robot.delta_with none \
  --robot.cameras "{ observation.images.image_top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, observation.images.image_left: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, observation.images.image_right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30} }"
  