"""
Example:
conda activate lerobot
python -m lerobot.robots.bi_piper.openpi_client \
  --host "192.168.10.1" \
  --port 8000 \
  --task "fold the towel" \
  --frequency 30 \
  --action_type absolute \
  --robot.type bi_piper \
  --robot.can_left can_left \
  --robot.can_right can_right \
  --robot.velocity 80 \
  --robot.init_type joint \
  --robot.delta_with none \
  --robot.cameras "{ observation.images.image_top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, observation.images.image_left: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, observation.images.image_right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30} }"

  """

from dataclasses import dataclass, field
from typing import List
import importlib
import numpy as np
import os
import time
import threading
import traceback
import imageio

import draccus

from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.async_inference.helpers import get_logger

# Import camera configs to register them with draccus.ChoiceRegistry
import lerobot.cameras.opencv.configuration_opencv  # noqa: F401
import lerobot.cameras.realsense.configuration_realsense  # noqa: F401

if importlib.util.find_spec("openpi_client") is None:
    raise ImportError("openpi_client is not installed. Please install it via `pip install openpi-client`.")

from openpi_client.websocket_client_policy import WebsocketClientPolicy


@dataclass
class OpenPIBiPiperClientConfig:
    robot: RobotConfig
    host: str = "localhost"
    port: int = 8000
    task: str = "do something"
    frequency: int = 10
    action_type: str = "absolute" # Server outputs are treated as absolute targets.
    # If set (or left as default), the client will move the robot to match the first
    # row's `observation.state` in that parquet before starting inference.
    init_state_parquet_path: str = "/home/agilex/.why/lerobot/data/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_left_0511/data/chunk-000/episode_000000.parquet"
    gripper_input_min: float = 0.0
    gripper_input_max: float = 0.76
    gripper_output_max_m: float = 0.07
    gripper_close_threshold: float = 0.01
    freeze_left_arm_at_first_frame: bool = True
    
    result_dir: str = "results/"
    camera_keys: List[str] = field(
        default_factory=lambda: [
            "observation.images.image_top",
            "observation.images.image_left",
            "observation.images.image_right",
        ]
    )
    fps: int = 10


class _VideoRecorder:
    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = save_dir
        self.fps = fps
        self._frames: List[np.ndarray] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, frame):
        if isinstance(frame, list):
            frame = np.concatenate(frame, axis=1)
        self._frames.append(frame)

    def save(self, task: str, success: bool):
        save_path = os.path.join(
            self.save_dir,
            f"{task.replace('.', '')}_{'success' if success else 'failed'}_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        imageio.mimwrite(save_path, self._frames, fps=self.fps)
        self._frames = []


class _KeyboardListener:
    def __init__(self):
        try:
            from sshkeyboard import listen_keyboard, stop_listening  # type: ignore
        except Exception:
            listen_keyboard = None
            stop_listening = None
        self._listen_keyboard = listen_keyboard
        self._stop_listening = stop_listening
        self._listener = (
            threading.Thread(target=self._listen_keyboard, args=(self._on_press,)) if self._listen_keyboard else None
        )
        if self._listener:
            self._listener.daemon = True
        self._quit = False
        self._success = None

    def listen(self):
        if self._listener:
            self._listener.start()

    def reset(self):
        self._quit = False
        self._success = None

    def _on_press(self, key):
        if key == "q":
            self._quit = True
        elif key == "y":
            self._success = True
            if self._stop_listening:
                self._stop_listening()
        elif key == "n":
            self._success = False
            if self._stop_listening:
                self._stop_listening()


class OpenPIBiPiperClient:
    def __init__(self, config: OpenPIBiPiperClientConfig):
        self.config = config
        self.logger = get_logger("openpi_bipiper_client")
        self.joint_action_keys = [
            "left_joint_1_pos", "left_joint_2_pos", "left_joint_3_pos",
            "left_joint_4_pos", "left_joint_5_pos", "left_joint_6_pos",
            "left_gripper_pos",
            "right_joint_1_pos", "right_joint_2_pos", "right_joint_3_pos",
            "right_joint_4_pos", "right_joint_5_pos", "right_joint_6_pos",
            "right_gripper_pos",
        ]
        self.policy_image_keys = {
            "observation.images.cam_high": "cam_high",
            "observation.images.image_top": "cam_high",
            "observation.images.cam_left_wrist": "cam_left_wrist",
            "observation.images.image_left": "cam_left_wrist",
            "observation.images.cam_right_wrist": "cam_right_wrist",
            "observation.images.image_right": "cam_right_wrist",
        }
        self.step_idx = 0
        self.frozen_left_arm_target: np.ndarray | None = None

        self.video_recorder = _VideoRecorder(config.result_dir, fps=config.fps)
        self.keyboard_listener = _KeyboardListener()

        self.robot = make_robot_from_config(config.robot)
        self.logger.info(f"Initialized robot: {self.robot.name}")

        self._is_finished = False
        self._server_metadata: dict | None = None
        self._stopped = False

    def start(self):
        self.keyboard_listener.listen()
        self.logger.info("Starting OpenPI BiPiper client...")
        # 1. 连接机器人硬件
        self.robot.connect()
        self._validate_joint_robot()
        
        # 2. 将机械臂移动到数据集首帧的 state，确保 delta 推理的基准一致
        self._move_robot_to_parquet_first_state(self.config.init_state_parquet_path)

        # 3. 再连接并初始化 OpenPI 服务端策略客户端
        # Monkey patch websockets.sync.client.connect to disable keepalive pings
        # This prevents the client from dropping the connection when OpenPI server
        # is busy compiling the model for the first time.
        import websockets.sync.client
        original_connect = websockets.sync.client.connect
        
        def patched_connect(*args, **kwargs):
            kwargs['ping_interval'] = None
            kwargs['ping_timeout'] = None
            kwargs.setdefault("open_timeout", 2.0)
            return original_connect(*args, **kwargs)
            
        websockets.sync.client.connect = patched_connect
        
        self.policy = WebsocketClientPolicy(self.config.host, self.config.port)
        
        # Restore original connect just in case
        websockets.sync.client.connect = original_connect
        
        try:
            self._server_metadata = self.policy.get_server_metadata()
            if isinstance(self._server_metadata, dict):
                self.logger.info(
                    "OpenPI server metadata keys: " + ", ".join(sorted(self._server_metadata.keys()))
                )
        except Exception:
            self._server_metadata = None

        self.logger.info(f"Connected to OpenPI server at {self.config.host}:{self.config.port}")

    def _reset_policy_connection(self) -> None:
        import websockets.sync.client

        try:
            ws = getattr(self.policy, "_ws", None)
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception:
            pass

        original_connect = websockets.sync.client.connect

        def patched_connect(*args, **kwargs):
            kwargs['ping_interval'] = None
            kwargs['ping_timeout'] = None
            kwargs.setdefault("open_timeout", 2.0)
            return original_connect(*args, **kwargs)

        websockets.sync.client.connect = patched_connect
        last_err: Exception | None = None
        try:
            for _ in range(20):
                try:
                    self.policy = WebsocketClientPolicy(self.config.host, self.config.port)
                    try:
                        self._server_metadata = self.policy.get_server_metadata()
                    except Exception:
                        self._server_metadata = None
                    return
                except Exception as e:
                    last_err = e
                    time.sleep(0.2)
        finally:
            websockets.sync.client.connect = original_connect
        if last_err is not None:
            if isinstance(last_err, ConnectionRefusedError):
                raise RuntimeError(
                    "OpenPI server refused connection during reconnect. "
                    "This usually means the server process crashed (e.g., XLA/ptxas compile failure) "
                    "or is still restarting."
                ) from last_err
            raise last_err

    def _infer_with_reconnect(self, obs: dict) -> dict:
        import websockets

        last_err: Exception | None = None
        for attempt in range(2):
            try:
                return self.policy.infer(obs)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                last_err = e
                self.logger.error(f"WebSocket connection error during infer (attempt {attempt + 1}/2): {e}")
                if attempt == 0:
                    self.logger.info("Reconnecting to OpenPI server...")
                    self._reset_policy_connection()
                    continue
                raise
            except Exception as e:
                last_err = e
                raise
        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenPI inference failed for unknown reasons.")

    def _move_robot_to_parquet_first_state(self, parquet_path: str) -> None:
        if not parquet_path:
            self.logger.info("init_state_parquet_path is empty; skipping init-state alignment.")
            return

        if not os.path.exists(parquet_path):
            self.logger.warning(f"init_state_parquet_path not found: {parquet_path}. Skipping init-state alignment.")
            # Still initialize internal caches
            self.robot.get_observation()
            return

        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("Reading parquet requires pandas. Please `pip install pandas pyarrow`.") from e

        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            self.logger.warning(f"Parquet has no rows: {parquet_path}. Skipping init-state alignment.")
            self.robot.get_observation()
            return

        state = np.array(df.iloc[0]["observation.state"], dtype=np.float64).reshape(-1)
        if len(state) != 14:
            raise ValueError(f"Expected 14-d `observation.state` in parquet, got {len(state)} from {parquet_path}")

        # Parquet state uses right7 + left7 ordering. Convert to robot command order left7 + right7.
        # Keep the joint pose from the first frame, but force both grippers fully open at init.
        right = state[:7].copy()
        left = state[7:].copy()
        right[6] = self.config.gripper_output_max_m
        left[6] = self.config.gripper_output_max_m
        target = np.concatenate([left, right], axis=0)
        self.frozen_left_arm_target = left.astype(np.float64)

        self.logger.info(
            f"Aligning robot initial joint state to parquet first frame: {parquet_path} "
            f"(state right7+left7 = {state.tolist()}, init grippers forced open to {self.config.gripper_output_max_m:.4f} m)"
        )

        init_action = {key: float(target[i]) for i, key in enumerate(self.joint_action_keys)}

        # Send a few times to reduce CAN packet loss, then wait briefly.
        for _ in range(3):
            self.robot.send_action(init_action)
            time.sleep(0.05)

        self.logger.info("Waiting for robot to reach parquet initial position...")
        time.sleep(2.0)
        self.logger.info("Robot init-state alignment done.")

        # Refresh observation so delta baseline uses the aligned pose.
        self.robot.get_observation()

    def control_loop(self):
        while not self._is_finished:
            # 拿到机器人的当前观测数据
            raw_obs = self.robot.get_observation()
            joint_obs = dict(raw_obs)
            obs = self._prepare_observation(dict(raw_obs))
            self.logger.info(f"Prompt: {obs['prompt']}")
            self.logger.info("Sending obs keys: " + ", ".join(sorted(obs.keys())))
            if self.step_idx == 0:
                try:
                    im = obs.get("images", {})
                    if isinstance(im, dict):
                        for k, v in im.items():
                            if hasattr(v, "shape"):
                                self.logger.info(f"Sending images[{k}] shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
                except Exception as e:
                    self.logger.warning(f"Failed to log image shapes: {e}")
            # Send observation to OpenPI server
            try:
                result = self._infer_with_reconnect(obs)
            except Exception:
                self._is_finished = True
                raise
            actions = result.get("actions", result.get("action")) # Support both "actions" and "action" keys
            
            if actions is None:
                self.logger.error(f"Server returned invalid result: {result.keys()}")
                continue
                
            # We typically get a sequence of actions (e.g. shape [chunk_size, action_dim])
            # Let's take the first action for immediate execution
            # Or if the server returns a single action [action_dim], we handle it
            action_seq = np.array(actions)
            if action_seq.ndim > 1:
                action = action_seq[0]
            else:
                action = action_seq

            action = self._prepare_action(action, joint_obs)
            self.robot.send_action(action)
            self._after_action()
            time.sleep(1 / self.config.frequency)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.logger.info("Stopping client...")
        try:
            self.robot.disconnect()
        except Exception as e:
            self.logger.warning(f"Robot disconnect failed (ignored): {e}")

    def _validate_joint_robot(self) -> None:
        robot_action_keys = list(self.robot.action_features.keys())
        if robot_action_keys != self.joint_action_keys:
            raise ValueError(
                "Joint inference requires `--robot.type bi_piper`. "
                f"Current action features are {robot_action_keys}."
            )

    def _map_dataset_gripper_to_piper(self, value: float) -> float:
        in_min = self.config.gripper_input_min
        in_max = self.config.gripper_input_max
        out_max = self.config.gripper_output_max_m
        if in_max <= in_min:
            raise ValueError("gripper_input_max must be larger than gripper_input_min")

        value = float(value)
        if value < float(self.config.gripper_close_threshold):
            return 0.0

        clipped = float(np.clip(value, in_min, in_max))
        normalized = (clipped - in_min) / (in_max - in_min)
        return float(normalized * out_max)

    def _build_policy_state(self, raw_obs: dict) -> list[float]:
        # The dataset/server use a 14-dim joint-only state in right7 + left7 order.
        return [
            float(raw_obs["right_joint_1_pos"]),
            float(raw_obs["right_joint_2_pos"]),
            float(raw_obs["right_joint_3_pos"]),
            float(raw_obs["right_joint_4_pos"]),
            float(raw_obs["right_joint_5_pos"]),
            float(raw_obs["right_joint_6_pos"]),
            float(raw_obs["right_gripper_pos"]),
            float(raw_obs["left_joint_1_pos"]),
            float(raw_obs["left_joint_2_pos"]),
            float(raw_obs["left_joint_3_pos"]),
            float(raw_obs["left_joint_4_pos"]),
            float(raw_obs["left_joint_5_pos"]),
            float(raw_obs["left_joint_6_pos"]),
            float(raw_obs["left_gripper_pos"]),
        ]

    def _extract_joint_action(self, action: np.ndarray) -> np.ndarray:
        action = np.array(action, dtype=np.float64).reshape(-1)
        if len(action) != 14:
            raise ValueError(f"Expected 14-d joint action from server, got {len(action)}.")

        # Server outputs are interpreted as right7 + left7 absolute targets.
        right = action[:7].astype(np.float64)
        left = action[7:].astype(np.float64)
        right[6] = self._map_dataset_gripper_to_piper(float(right[6]))
        left[6] = self._map_dataset_gripper_to_piper(float(left[6]))
        return np.concatenate([left, right], axis=0)

    def _get_current_joint_state(self, raw_obs: dict) -> np.ndarray:
        return np.array([float(raw_obs[key]) for key in self.joint_action_keys], dtype=np.float64)

    def _ensure_chw_uint8(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # HWC -> CHW (what aloha_policy expects: "c h w -> h w c" conversion)
        if img.ndim == 3 and img.shape[-1] in [1, 3, 4] and img.shape[0] > 4 and img.shape[1] > 4:
            img = np.transpose(img, (2, 0, 1))

        if img.ndim != 3:
            raise ValueError(f"Expected image with 3 dims (CHW), got shape={img.shape}")
        if img.shape[0] not in [1, 3, 4]:
            raise ValueError(f"Expected image channels in [1,3,4], got shape={img.shape}")

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return np.ascontiguousarray(img)

    def _prepare_observation(self, observation: dict) -> dict:
        state = self._build_policy_state(observation)

        for key in self.robot._motors_ft.keys():
            if key not in observation:
                raise KeyError(f"Expected key {key} in observation, but got {observation.keys()}")
            observation.pop(key)

        out: dict = {"prompt": self.config.task, "state": state}

        keys_to_remove = []
        processed_images: dict[str, np.ndarray] = {}
        for key, value in observation.items():
            if key not in self.policy_image_keys:
                continue

            image_name = self.policy_image_keys[key]
            if hasattr(value, "cpu"):
                value = value.detach().cpu().numpy()

            if isinstance(value, np.ndarray):
                get_logger("openpi_client").info(
                    f"===> Camera [{image_name}] raw shape before process: {value.shape}, dtype: {value.dtype}"
                )
                value = np.squeeze(value)
                if value.ndim == 2:
                    value = np.expand_dims(value, axis=-1)
                if value.ndim == 3 and value.shape[0] in [1, 3]:
                    value = np.transpose(value, (1, 2, 0))

                import cv2

                if value.ndim == 3 and value.shape[2] == 1:
                    value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
                value = cv2.resize(value, (224, 224))
                if value.ndim == 3 and value.shape[0] in [1, 3, 4] and value.shape[1] == 224 and value.shape[2] == 224:
                    value = np.transpose(value, (1, 2, 0))
                if value.ndim == 3 and value.shape[2] == 3:
                    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                if value.dtype != np.uint8:
                    if value.max() <= 1.0:
                        value = (value * 255).astype(np.uint8)
                    else:
                        value = value.astype(np.uint8)

                get_logger("openpi_client").info(
                    f"===> Camera [{image_name}] final shape sent to OpenPI: {value.shape}, dtype: {value.dtype}"
                )

            processed_images[image_name] = value
            keys_to_remove.append(key)

        for key in keys_to_remove:
            observation.pop(key)

        if "cam_low" not in processed_images and "cam_high" in processed_images:
            self.logger.warning("Camera `cam_low` is missing; reusing `cam_high` as fallback.")
            processed_images["cam_low"] = processed_images["cam_high"]

        processed_images = {k: self._ensure_chw_uint8(v) for k, v in processed_images.items()}
        out["images"] = processed_images

        return out

    def _prepare_action(self, action, _raw_obs: dict) -> dict:
        raw_action = np.array(action, dtype=np.float64).reshape(-1)
        self.logger.info(
            f"[Frame {self.step_idx}] Server raw action ({len(raw_action)}): "
            + ", ".join(f"{x:.4f}" for x in raw_action)
        )
        if len(raw_action) >= 14:
            self.logger.info(
                f"[Frame {self.step_idx}] Server raw grippers (right@7, left@14): "
                f"{raw_action[6]:.4f}, {raw_action[13]:.4f}"
            )

        action = self._extract_joint_action(action)

        if self.config.freeze_left_arm_at_first_frame and self.frozen_left_arm_target is not None:
            action[:7] = self.frozen_left_arm_target
            self.logger.info(
                "Left arm is frozen at parquet first-frame target: "
                + ", ".join(f"{x:.4f}" for x in self.frozen_left_arm_target)
            )

        if self.config.action_type == "delta":
            self.logger.warning(
                "Server outputs are configured as absolute targets; delta accumulation is disabled. "
                "Use `--action_type absolute` to silence this warning."
            )

        # 夹爪限位保护，单位为米
        action[6] = float(np.clip(action[6], 0.0, self.config.gripper_output_max_m))
        action[13] = float(np.clip(action[13], 0.0, self.config.gripper_output_max_m))

        left_arm_str = ", ".join([f"{x:.3f}" for x in action[0:6]])
        left_gripper_str = f"{action[6]:.3f}"
        right_arm_str = ", ".join([f"{x:.3f}" for x in action[7:13]])
        right_gripper_str = f"{action[13]:.3f}"

        self.logger.info("====> ACTION COMMAND TO BIPIPER <====")
        self.logger.info(f"Left  Arm (Joint 1-6): [{left_arm_str}] | Gripper(m): {left_gripper_str}")
        self.logger.info(f"Right Arm (Joint 1-6): [{right_arm_str}] | Gripper(m): {right_gripper_str}")
        self.logger.info("=====================================")
        self.step_idx += 1

        return {key: action[i].item() for i, key in enumerate(self.joint_action_keys)}

    def _after_action(self):
        obs = self.robot.get_observation()
        # 原本记录视频是去查 'observation.images.xxx'
        frames = [obs[key] for key in self.config.camera_keys if key in obs]
        if frames:
            self.video_recorder.add(frames)
        if self.keyboard_listener._quit:
            while self.keyboard_listener._success is None:
                time.sleep(0.1)
            self.video_recorder.save(task=self.config.task, success=self.keyboard_listener._success)
            self._is_finished = True


@draccus.wrap()
def main(cfg: OpenPIBiPiperClientConfig):
    client = OpenPIBiPiperClient(cfg)
    client.start()
    try:
        client.control_loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        client.logger.error(f"Error in control loop: {e}")
        client.logger.error(traceback.format_exc())
    finally:
        client.stop()


if __name__ == "__main__":
    main()
