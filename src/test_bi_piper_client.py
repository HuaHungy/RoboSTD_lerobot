import time
import json
import zmq
from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.configuration_bi_piper import BiPiperConfig

def run_client():
    """
    LeRobot 作为客户端 (订阅者 Subscriber)
    它的作用是接收网络中服务端发来的数据，并转换为 action 发送给底层硬件。
    """
    print("1. 初始化双臂配置...")
    config = BiPiperConfig(can_left="can0", can_right="can1", velocity=30)
    robot = BiPiper(config)
    
    print("2. 尝试连接左右机械臂...")
    try:
        robot.connect()
        print("✅ 机械臂连接成功！")
        
        print("3. 初始化网络客户端 (ZMQ SUB)...")
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        
        # 【关键】CONFLATE 参数用于只保留最新的一条消息。
        # 如果机器人执行动作比服务端发得慢，它会丢弃积压的旧指令，避免巨大的延迟。
        socket.setsockopt(zmq.CONFLATE, 1)  
        
        # 连接到服务端的 IP，如果是本机则是 localhost。替换为你实际的 Server IP
        socket.connect("tcp://localhost:5555")  
        socket.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅所有消息
        socket.setsockopt(zmq.RCVTIMEO, 1000)       # 接收超时设置为1秒，防止死锁
        
        print("✅ 客户端网络已连接，正在监听服务端 (tcp://localhost:5555) 的指令...")
        print("💡 按下 Ctrl+C 即可安全停止")

        while True:
            try:
                # 4. 接收网络数据
                message_str = socket.recv_string()
                data = json.loads(message_str)
                
                left_state = data.get("left_state")
                right_state = data.get("right_state")
                
                # 如果没有收到完整的左右臂数据，则跳过本次循环
                if not left_state or not right_state:
                    continue

                # 5. 构造 LeRobot 标准的 action 字典
                action = {}
                for i, name in enumerate(config.joint_names):
                    action[f"left_{name}_pos"] = left_state[i]
                    action[f"right_{name}_pos"] = right_state[i]
                
                # 6. 调用框架接口，将指令发送给硬件
                robot.send_action(action)
                
            except zmq.error.Again:
                # 如果超过 1000ms 没有收到数据，会走到这里
                print("⏳ 等待服务端数据超时，继续重试...")
                continue
                
    except KeyboardInterrupt:
        print("\n⏹️ 收到停止信号。")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("正在断开网络和硬件连接...")
        try:
            socket.close()
            context.term()
        except:
            pass
        robot.disconnect()
        print("✅ 连接已安全断开。")

if __name__ == "__main__":
    run_client()
