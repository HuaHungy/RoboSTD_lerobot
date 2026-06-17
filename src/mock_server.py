import zmq
import time
import json
import math

def run_server():
    """
    这是一个模拟的 ZMQ 服务端 (发布者 Publisher)
    它的作用是生成关节指令数据，并通过网络广播出去。
    在实际场景中，这里可能是遥操作的主臂数据采集程序，或者是 AI 模型的推理输出。
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # 绑定到 5555 端口
    socket.bind("tcp://*:5555")
    print("✅ 服务端已启动，正在 tcp://*:5555 广播关节数据...")
    print("提示: 这是一个模拟服务端，正在发送微小幅度的正弦波运动指令...")

    t = 0.0
    try:
        while True:
            # 模拟生成双臂的关节数据 (这里用正弦波模拟平滑运动)
            angle = math.sin(t) * 0.1  # 幅度在 -0.1 到 0.1 弧度之间摆动
            
            # 构造左右臂的 7 个自由度目标状态 (前6个弧度，最后一个米)
            target_left = [0.0, angle, -angle, angle, -angle, 0.0, 0.06]
            target_right = [0.0, angle, -angle, angle, -angle, 0.0, 0.06]
            
            # 封装为 JSON 格式的消息
            message = {
                "timestamp": time.time(),
                "left_state": target_left,
                "right_state": target_right
            }
            
            # 通过 ZMQ 发送出去
            socket.send_string(json.dumps(message))
            
            t += 0.1
            time.sleep(0.05)  # 发送频率 20Hz
            
    except KeyboardInterrupt:
        print("\n⏹️ 收到停止信号，服务端已停止。")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    run_server()
