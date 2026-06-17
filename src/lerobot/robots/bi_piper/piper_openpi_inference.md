**概览**
- 本文介绍如何在 LeRobot 中使用通用客户端收发方式连接远端 OpenPI 推理服务端，并驱动 Piper 双臂机器人（BiPiper）。同时总结其他机器人在 LeRobot 中的推理实现范式，并提供接入任意新机器人的通用方法。
- 关键参考代码：
  - BiPiper 机器人实现：[bi_piper.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/bi_piper.py)，[bi_piper_end_effector.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/bi_piper_end_effector.py)
  - 双臂基类（观测/动作拼接、相机管理）：[bi_base_robot.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_base_robot/bi_base_robot.py)
  - OpenPI 客户端（本文新增）：[openpi_client.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/openpi_client.py)
  - 异步推理通用客户端/服务器（gRPC）：[robot_client.py](file:///home/agilex/lerobot/src/lerobot/async_inference/robot_client.py)，[policy_server.py](file:///home/agilex/lerobot/src/lerobot/async_inference/policy_server.py)
  - OpenPI 模型集成（Pi0/Pi0.5）：[modeling_pi0.py](file:///home/agilex/lerobot/src/lerobot/policies/pi0/modeling_pi0.py)，[processor_pi0.py](file:///home/agilex/lerobot/src/lerobot/policies/pi0/processor_pi0.py)
  - 参考的 OpenPI WebSocket 客户端（另一仓库）：[robot_client_openpi.py](file:///home/agilex/RoboCoin-lerobot/src/lerobot/scripts/server/robot_client_openpi.py)

**体系结构**
- 服务端：OpenPI 模型在远端设备运行，提供 WebSocket 推理接口（openpi-client 封装）。
- 客户端：运行在接入机器人侧，完成
  - 从机器人采集观测（关节/末端状态 + 多路相机画面）
  - 整理成 OpenPI 期望的观测字典（包含 observation.state、observation.images.*、prompt）
  - 通过 WebSocket 调用 infer(obs) 获取动作序列
  - 将动作序列逐条映射到机器人动作空间并执行
- 关键数据契约
  - 观测：observation.state 为一维数组（按机器人 action_features 的顺序拼接），图像键名为 observation.images.<camera_name>，自然语言指令为 prompt
  - 动作：序列化为列表[List[float]]；客户端映射成 {feature_name: value} 字典传入 robot.send_action

**其他机器人推理实现范式**
- gRPC 异步推理（内置）
  - 协议见 [services.proto](file:///home/agilex/lerobot/src/lerobot/transport/services.proto)，服务器端见 [policy_server.py](file:///home/agilex/lerobot/src/lerobot/async_inference/policy_server.py)，客户端见 [robot_client.py](file:///home/agilex/lerobot/src/lerobot/async_inference/robot_client.py)。
  - 特点：观测/动作分片、动作队列聚合、must_go 节拍控制、重试/延迟统计；适合统一部署自托管模型。
- ZMQ/自定义套接字
  - Unitree G1 通过 ZeroMQ 桥接硬件接口（示例：[unitree_sdk2_socket.py](file:///home/agilex/lerobot/src/lerobot/robots/unitree_g1/unitree_sdk2_socket.py)）。
  - LeKiwi 客户端使用 ZMQ 与远端底盘/相机交互（示例：[lekiwi_client.py](file:///home/agilex/lerobot/src/lerobot/robots/lekiwi/lekiwi_client.py)）。
- 本文新增的 OpenPI WebSocket 模式
  - 适用于远端已有的 OpenPI 推理服务；无需在本机加载模型，仅需安装 openpi-client。

**Piper（BiPiper）使用指南**
- 依赖
  - 机器人 SDK：piper_sdk（机器人侧控制，见 [piper.py](file:///home/agilex/lerobot/src/lerobot/robots/piper/piper.py)）
  - OpenPI 客户端：openpi-client（WebSocket 调用）
  - 可选：draccus（CLI 参数解析）
- 连接与观测/动作
  - BiPiper 将左右臂各 7 维（6 关节/位姿 + 1 夹爪）按前缀 left_/right_ 拼接为 observation_features 与 action_features（见 [bi_base_robot.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_base_robot/bi_base_robot.py#L82-L99)）。
  - 客户端将 robot.get_observation() 中的电机相关键提取并拼接为 observation.state；其余图像键保留为 observation.images.*。
  - WebSocket 返回的动作序列按 action_features 的顺序映射为字典后调用 robot.send_action。
- 运行示例

```bash
python -m lerobot.robots.bi_piper.openpi_client \
  --host "192.168.1.20" \
  --port 18000 \
  --task "fold the towel" \
  --frequency 10 \
  --robot.type bi_piper_end_effector \
  --robot.can_left can0 \
  --robot.can_right can1 \
  --robot.velocity 30 \
  --robot.init_type joint \
  --robot.delta_with previous \
  --robot.pose_units "[001mm, 001mm, 001mm, 001degree, 001degree, 001degree, 001mm]" \
  --robot.model_pose_units "[m, m, m, radian, radian, radian, m]" \
  --robot.cameras "{ observation.images.cam_high: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, observation.images.cam_right_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }"
```

- 参数要点
  - host/port：OpenPI 服务端地址（可在另一台可达的设备上运行）
  - task：当前任务的自然语言描述
  - robot.type：选择 bi_piper 或 bi_piper_end_effector
  - cameras：设置相机来源与分辨率，键名需形如 observation.images.<name>
- 运行流程
  - 客户端启动后连接机器人与相机 → 采集观测 → 发送至 OpenPI → 接收一段动作序列 → 逐条执行并按 frequency 控速
  - 夹爪安全处理：对每侧第 7 维夹爪值做简单裁剪（<300 视为闭合、>1000 上限），可按硬件微调（见 [openpi_client.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/openpi_client.py) 中 _prepare_action）

**如何接入其他机器人（通用方法）**
- 实现 Robot 接口
  - 新增机器人需继承 [Robot](file:///home/agilex/lerobot/src/lerobot/robots/robot.py) 或对应基类（如 BaseRobot/BiBaseRobot）并实现：
    - 连接/断开：connect()/disconnect()
    - 观测：get_observation() 返回 {<motor_keys>: float, <camera_keys>: np.ndarray}
    - 动作：send_action(dict[str, float])，与 action_features 对齐
    - 状态：get_joint_state()/set_joint_state() 或 get_ee_state()/set_ee_state()
  - 确认 observation_features 与 action_features 定义合理（键顺序稳定、单位一致）。
- 复用 OpenPI 客户端
  - 若按 LeRobot 约定返回观测/动作，上述 [openpi_client.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/openpi_client.py) 可直接复用，只需通过 --robot.type 指定你的新机器人类型，并在配置中添加相机与单位映射。
  - 若需特殊处理（如夹爪阈值、速度限制），可在 _prepare_action 中按机器人特性裁剪。
- 单位与安全
  - 确保模型期望单位（通常是 radian/m）与硬件 SDK 单位一致，必要时在 send_action 内做转换（参考 Piper 中 0.001 度/毫米与弧度/米的映射：[configuration_bi_piper.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/configuration_bi_piper.py)）。
  - 对增量控制时做目标差值限幅（可参考 [ensure_safe_goal_position](file:///home/agilex/lerobot/src/lerobot/robots/utils.py#L82-L118)）。
- 网络与性能
  - 网络：OpenPI WebSocket 模式对网络稳定性要求高，建议内网直连；如需更强鲁棒性可考虑切换至 gRPC 异步方案。
  - 速率：根据机械臂与相机性能调节 frequency 与相机 FPS，避免控制环堵塞。

**常见问题**
- openpi_client 未安装：按提示执行 pip install openpi-client
- piper_sdk 未安装或未连接：确认 CAN 口、供电、驱动安装；参见 [piper.py](file:///home/agilex/lerobot/src/lerobot/robots/piper/piper.py)
- 图像键名不匹配：确保相机配置的键名以 observation.images. 开头，且与 OpenPI 模型预处理一致
- 动作维度不一致：检查 robot.action_features 的键数量与顺序，必要时在 _prepare_action 中重排

**附：代码走读（关键路径）**
- 机器人侧
  - 双臂拼接与 IO：[bi_base_robot.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_base_robot/bi_base_robot.py)
  - Piper SDK 调用：[piper.py](file:///home/agilex/lerobot/src/lerobot/robots/piper/piper.py)
- 客户端侧
  - OpenPI WebSocket 客户端：[openpi_client.py](file:///home/agilex/lerobot/src/lerobot/robots/bi_piper/openpi_client.py)
  - gRPC 通用客户端：[robot_client.py](file:///home/agilex/lerobot/src/lerobot/async_inference/robot_client.py)
- 参考实现
  - 外部仓库的 OpenPI 客户端范式：[robot_client_openpi.py](file:///home/agilex/RoboCoin-lerobot/src/lerobot/scripts/server/robot_client_openpi.py)

