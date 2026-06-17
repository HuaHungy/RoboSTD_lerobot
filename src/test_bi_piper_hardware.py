import time
import numpy as np
from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.configuration_bi_piper import BiPiperConfig

def move_two_arms_to_target():
    print("初始化双臂配置...")
    # velocity提高到50，以保证机械臂能有足够的运动速度
    config = BiPiperConfig(can_left="can_left", can_right="can_right", velocity=40)
    robot = BiPiper(config)
    
    print("尝试连接左右机械臂...")
    try:
        robot.connect()
        print("✅ 机械臂连接成功！")
        
        # 读取当前状态
        obs = robot.get_observation()
        print(f"✅ 当前双臂状态读取成功: \n{obs}")
        
        # 构造目标位置
        # 这里为了演示，我们让左右臂稍微抬起一点，张开夹爪
        # 7个关节：[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
        # 单位: 前6个旋转关节为 radian (弧度), 最后的夹爪为 m (米)
        
        # 设定一个特定的位置：大臂（关节2或3）微微抬起，约 0.5 弧度 (30度左右)
        # 注意：这里的正负号取决于 Piper 的实际物理坐标系定义，我们先试着改变 joint_2 和 joint_3
        # target_left_state = [-0.0, 0.6, -0.7, -0.047, 1.082, 0.0206, 0.7265]
        # target_right_state = [0.0, 0.6, -0.7, -0.0071, 1.0737, 0.0267, 0.746]
        # target_left_state = [-0.0, 0.6, -0.7, -0.047, 1.082, 0.0206, 0.7265]
        # target_right_state = [0.164, 1.349, -1.036, 0.525, 0.798, -1.039, 0.746]
        target_left_state = [-0.164, 1.349, -1.036, -0.525, 0.9, 1.039, 0.746]
        target_right_state = [0.164, 1.349, -1.036, 0.525, 0.798, -1.039, 0.746]
        # target_left_state = [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.00]
        # target_right_state = [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.00]
        print("➡️ 准备进行平滑的轨迹插补，移动到抬升目标位置...")
        
        # 提取当前状态
        current_left_state = [obs[f"left_{name}_pos"] for name in config.joint_names]
        current_right_state = [obs[f"right_{name}_pos"] for name in config.joint_names]
        
        # 设定插补步数，减少到 40 步，总耗时约 2 秒，加快整体速度
        steps = 1
        
        for step in range(1, steps + 1):
            action = {}
            # 计算当前步的插值位置
            for i, name in enumerate(config.joint_names):
                # 线性插值
                left_pos = current_left_state[i] + (target_left_state[i] - current_left_state[i]) * (step / steps)
                right_pos = current_right_state[i] + (target_right_state[i] - current_right_state[i]) * (step / steps)
                
                action[f"left_{name}_pos"] = left_pos
                action[f"right_{name}_pos"] = right_pos
            
            # 为了防止单次CAN丢包，每一步连续发送2次
            robot.send_action(action)
            robot.send_action(action)
            
            # 缩短等待时间 (40步 * 0.05秒 = 2秒总时长)，使运动更加连贯流畅
            time.sleep(0.05)
            
            if step % 10 == 0:
                print(f"   已发送插补进度: {step}/{steps}...")
                
        print("✅ 发送动作指令成功！")
        
        # 等待机械臂平滑运动到位
        print("⏳ 正在运动，请等待让其完全停止...")
        for i in range(3):
            time.sleep(1)
            print(f"   等待中... {3-i}s")
            
        # 再次读取状态，验证是否到达目标位置
        obs_after = robot.get_observation()
        print(f"✅ 运动后当前状态读取成功: \n{obs_after}")
        
    except Exception as e:
        print(f"❌ 通信或运动过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("✅ 连接已安全断开。")

if __name__ == "__main__":
    move_two_arms_to_target()
