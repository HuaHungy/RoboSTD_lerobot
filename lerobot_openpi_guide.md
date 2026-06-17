# LeRobot 与 OpenPI 推理服务端接入指南

本文档旨在说明如何使用 LeRobot 框架的机器人（例如 Piper）作为客户端，通过 WebSocket 接入物理智能（Physical Intelligence）开源的 OpenPI 推理服务端，并为未来接入其他新型机器人提供通用的操作指南。

## 1. 架构概览

整个系统分为**客户端（Client）**和**服务端（Server）**两部分：
- **服务端 (OpenPI Server)**: 运行 `openpi.serving.websocket_policy_server`，负责加载预训练的视觉-语言-动作模型（如 Pi0, Pi0-FAST），通过 WebSocket 接收包含图像、状态和提示词的数据包，进行模型前向推理（JAX/XLA），并返回关节动作。
- **客户端 (LeRobot Client)**: 负责通过各类 SDK 与硬件通信（如 `bi_piper_end_effector`），周期性地读取机械臂的关节状态（State）和相机的实时画面（Images），并将其按照服务端所期望的格式打包发送。接收到服务端返回的动作（Actions）后，将其下发给硬件执行。

---

## 2. 核心数据通信格式

为了成功与 OpenPI 提供的策略（如 `aloha_policy` 或 `droid_policy`）通信，客户端必须将 LeRobot 原始的扁平化观测数据（Observation）转换为特定的层次化字典格式。

OpenPI (以 `aloha_policy` 为例) 期望接收的数据格式如下：

```python
{
    "images": {
        "cam_high": np.ndarray,         # 形状必须为 (C, H, W)，即 (3, 224, 224)，数据类型 uint8
        "cam_left_wrist": np.ndarray,   # 同上
        "cam_right_wrist": np.ndarray,  # 同上
        # 其他相机如果未提供，服务端会自动使用全 0 矩阵填充 (zero-padding)
    },
    "state": [float, ...],              # 包含 14 个浮点数的列表，代表双臂和夹爪的关节角度和位置
    "prompt": "do something"            # 字符串，自然语言指令
}
```

> **注意**：
> 1. **键名**：字典必须严格包含 `"images"`、`"state"` 和 `"prompt"`。不能使用 `"image"`，否则会触发服务端 Transforms 逻辑冲突导致维度报错。
> 2. **图像格式**：必须为 `(C, H, W)`。如果从 LeRobot `OpenCVCamera` 读取出来的形状是 `(H, W, C)`，必须通过 `np.transpose(img, (2, 0, 1))` 转换。
> 3. **图像尺寸**：为了避免服务端 OOM 或 Transforms 维度越界，客户端建议提前将图像 `cv2.resize` 到模型所需的 `(224, 224)` 或 `(256, 256)`。

---

## 3. 接入新机器人的通用步骤

如果未来需要接入全新的机器人（如 `my_new_robot`），请按照以下通用步骤操作：

### 步骤 1：确认 LeRobot 硬件配置
确保新机器人在 LeRobot 框架下已注册，并可以通过 `make_robot_from_config` 实例化。了解该机器人的：
- 关节数量（Action / State Dimension）
- 各个相机的名称标识（如 `observation.images.cam_high`）

### 步骤 2：创建 OpenPI 客户端脚本
参考 `src/lerobot/robots/bi_piper/openpi_client.py`，为新机器人创建一个客户端脚本。你需要重写 `_prepare_observation` 和 `_prepare_action` 两个核心方法：

#### 2.1 重写 `_prepare_observation`
从 LeRobot 获取的原始 `obs` 是一个扁平字典。你需要：
1. 提取所有 `observation.images.*` 的图像数据。
2. 对每个图像执行：`cv2.resize` (缩放至224) -> `np.transpose` (转为 CHW) -> 类型转换为 `uint8`。
3. 提取所有的关节状态数据，按模型训练时的顺序（如左臂、左夹爪、右臂、右夹爪）拼装成一个 Python `list`。
4. 将整理好的数据赋值给 `obs["images"]`、`obs["state"]` 和 `obs["prompt"]`，并**移除任何多余的键**（尤其是 `"image"`）。

#### 2.2 重写 `_prepare_action`
服务端返回的 `result["action"]` 是一个包含动作序列的 numpy 数组（例如形状为 `(action_horizon, action_dim)`）。
客户端通常只需要取第一个动作，或者客户端控制循环会自动遍历动作序列。
你需要将长度为 `action_dim` 的动作数组，映射回 LeRobot 所期望的字典格式（对应 `robot.action_features` 的键），同时可以加入必要的软件限位（Clip）保护硬件。

### 步骤 3：对齐服务端策略 (Policy)
OpenPI 默认提供 `aloha_policy`、`droid_policy` 等。如果你的新机器人的 `state` 维度不是 14，或者相机的机位不同，你需要在 OpenPI 服务端：
1. 新建或修改一个 `policy.py`（参考 `aloha_policy.py`）。
2. 在 `DataTransformFn` 中调整 `action_dim`，修改 `EXPECTED_CAMERAS` 列表。
3. 在 `_decode_state` 中实现正确的硬件空间映射（例如角度转换、符号反转）。

### 步骤 4：处理常见故障 (Troubleshooting)

- **`KeyError: 'images'` / `TypeError`**: 检查客户端发送的键是否被错误命名为 `"image"`，导致被服务端的 `ResizeImages` 错误拦截。
- **`einops.EinopsError: Wrong shape`**: 检查客户端发送的图像是否包含 Batch 维度，或者通道顺序是否是 `(H, W, C)`。必须保证发送的每一张图严格为 `(C, H, W)` 且为 3 维。
- **`ConnectionClosedError: no close frame received or sent`**: 通常发生在客户端发送第一帧后。这是因为服务端机器在进行 JAX 的首次 JIT 编译时（XLA Compiler）耗尽了系统的 Host RAM 或 GPU VRAM，导致进程被 Linux 系统的 OOM-Killer 强行杀死。解决办法是升级服务端内存，配置大容量 Swap，或减小模型的 Batch Size。

---
**编写者**: Trae AI 
**更新日期**: 2026-04-09
