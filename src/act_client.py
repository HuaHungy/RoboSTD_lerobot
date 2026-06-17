#!/usr/bin/env python3


"""
python src/act_client.py \
    --host 192.168.10.1 \
    --port 8765 \
    --frequency 30 \
    --robot.type bi_piper \
    --robot.can_left can_left \
    --robot.can_right can_right \
    --robot.velocity 30 \
    --robot.cameras "{observation.images.image_top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, observation.images.image_left: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, observation.images.image_right: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}"

    """
import dataclasses
import json
import time
import base64
import traceback
import os
import shutil
import threading
import sys
from typing import Dict, Any

import cv2
import numpy as np
import draccus
import websockets.sync.client as ws_client

from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
import lerobot.robots.bi_piper.configuration_bi_piper  # noqa: F401
import lerobot.cameras.opencv.configuration_opencv  # noqa: F401

@dataclasses.dataclass
class ACTClientConfig:
    robot: RobotConfig
    host: str = "localhost"
    port: int = 8765
    frequency: int = 30
    freeze_left_arm: bool = False
    ws_timeout_s: float = 2.0
    result_dir: str = "result"
    show_window: bool = True

    # Most training pipelines use RGB. OpenCV provides BGR images by default.
    bgr_to_rgb: bool = True
    jpeg_quality: int = 95  # 提高质量，减少压缩损失

    # If True, missing state keys will raise instead of silently defaulting to 0.
    strict_state_keys: bool = True

    # Your datasets store gripper values roughly in the 0~0.75 range.
    # Map that range into the robot command range 0~0.07m before send_action.
    gripper_input_min: float = 0.0
    gripper_input_max: float = 0.76
    gripper_output_max_m: float = 0.07
    gripper_close_threshold: float = 0.01

    # Print the raw 14-d action array from the server at every control step.
    log_server_action_every_step: bool = True

    # 相机顺序必须与训练时一致：left, right, top
    camera_mapping: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "observation.images.image_left": "left",
            "observation.images.image_right": "right",
            "observation.images.image_top": "top",
        }
    )


class _VideoWriters:
    def __init__(self, save_dir: str, fps: int):
        self.save_dir = save_dir
        self.fps = int(fps)
        os.makedirs(self.save_dir, exist_ok=True)
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._sizes: dict[str, tuple[int, int]] = {}

    def _fourcc(self) -> int:
        return cv2.VideoWriter_fourcc(*"mp4v")

    def write(self, name: str, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            return
        h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        size = (w, h)
        if name not in self._writers:
            path = os.path.join(self.save_dir, f"{name}.mp4")
            writer = cv2.VideoWriter(path, self._fourcc(), self.fps, size)
            self._writers[name] = writer
            self._sizes[name] = size
        else:
            expected = self._sizes[name]
            if size != expected:
                frame_bgr = cv2.resize(frame_bgr, expected)
        self._writers[name].write(frame_bgr)

    def close(self) -> None:
        for w in self._writers.values():
            try:
                w.release()
            except Exception:
                pass
        self._writers = {}
        self._sizes = {}


class ACTClient:
    def __init__(self, config: ACTClientConfig):
        self.config = config
        self.frequency = config.frequency
        self.freeze_left_arm = config.freeze_left_arm
        self.camera_mapping = config.camera_mapping
        self._run_time = time.strftime("%Y%m%d_%H%M%S")
        self._label: str | None = None
        self._staging_dir = os.path.join(self.config.result_dir, "_staging", self._run_time)

        if self.config.show_window and not os.environ.get("DISPLAY"):
            self.config.show_window = False
            print("⚠️ 未检测到 DISPLAY，已自动关闭窗口显示（仍会录制视频）。")
        self._window_name = "ACT Client"
        if self.config.show_window:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                try:
                    cv2.startWindowThread()
                except Exception:
                    pass
            except cv2.error as e:
                self.config.show_window = False
                print(f"⚠️ OpenCV 无 GUI 支持，已自动关闭窗口显示（仍会录制视频）。错误: {e}")

        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        print("✅ 机械臂已连接")

        self.ws = ws_client.connect(f"ws://{config.host}:{config.port}")
        print(f"✅ 已连接到 ACT 服务器 {config.host}:{config.port}")

        self.step_idx = 0
        self._is_finished = False
        self._warned_missing_state_keys = False
        self._writers = _VideoWriters(save_dir=self._staging_dir, fps=self.frequency)
        self._curve_left_grip_raw: list[float] = []
        self._curve_left_grip_cmd: list[float] = []
        self._curve_right_grip_raw: list[float] = []
        self._curve_right_grip_cmd: list[float] = []

        # 启动后台线程监听终端输入
        self._input_thread = threading.Thread(target=self._input_listener_thread, daemon=True)
        self._input_thread.start()
        print("✅ 终端输入监听已启动（可随时输入 c=success 或 f=fail）")

        # Dataset/action convention in your meta is right7 + left7.
        self._state_keys: list[str] = [
            "right_joint_1_pos", "right_joint_2_pos", "right_joint_3_pos",
            "right_joint_4_pos", "right_joint_5_pos", "right_joint_6_pos",
            "right_gripper_pos",
            "left_joint_1_pos", "left_joint_2_pos", "left_joint_3_pos",
            "left_joint_4_pos", "left_joint_5_pos", "left_joint_6_pos",
            "left_gripper_pos",
        ]
        self._left_state_keys: list[str] = [
            "left_joint_1_pos", "left_joint_2_pos", "left_joint_3_pos",
            "left_joint_4_pos", "left_joint_5_pos", "left_joint_6_pos",
            "left_gripper_pos",
        ]

    def _input_listener_thread(self):
        """后台线程：监听终端输入，用户可随时输入 c 或 f"""
        try:
            while not self._is_finished:
                try:
                    # 使用非阻塞的方式读取单个字符
                    user_input = input().strip().lower()
                    if user_input in ("c",):
                        print("📝 输入: c - 标记为 SUCCESS")
                        self._label = "success"
                        self._is_finished = True
                    elif user_input in ("f",):
                        print("📝 输入: f - 标记为 FAIL")
                        self._label = "fail"
                        self._is_finished = True
                except EOFError:
                    # 当 stdin 关闭时，退出线程
                    break
        except Exception as e:
            print(f"⚠️ 输入监听线程出错: {e}")

    def _to_native(self, val: Any):
        # Ensure numpy scalars / torch tensors can be JSON-serialized.
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        if isinstance(val, np.ndarray):
            return val.tolist()[0] if val.size == 1 else val.tolist()
        return val

    def _extract_state(self, obs: Dict[str, Any]) -> list[float]:
        missing = [k for k in self._state_keys if k not in obs]
        if missing:
            if self.config.strict_state_keys:
                raise KeyError(f"Observation missing state keys: {missing}")
            if not self._warned_missing_state_keys:
                print(f"⚠️ observation.state 缺少字段，将用 0.0 填充: {missing}")
                self._warned_missing_state_keys = True

        state: list[float] = []
        for k in self._state_keys:
            state.append(float(self._to_native(obs.get(k, 0.0))))
        return state

    def _map_gripper_action(self, raw_value: float) -> float:
        in_min = float(self.config.gripper_input_min)
        in_max = float(self.config.gripper_input_max)
        out_max = float(self.config.gripper_output_max_m)
        if in_max <= in_min:
            raise ValueError("gripper_input_max must be larger than gripper_input_min")

        # Treat very small raw gripper values as a fully closed command to improve grasp closure.
        if raw_value < float(self.config.gripper_close_threshold):
            return 0.0

        clipped = float(np.clip(raw_value, in_min, in_max))
        normalized = (clipped - in_min) / (in_max - in_min)
        return float(normalized * out_max)

    def _encode_image(self, img: np.ndarray) -> str:
        if img is None:
            return ""
        if not isinstance(img, np.ndarray):
            return ""

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # OpenCV images are BGR; many models were trained on RGB.
        if self.config.bgr_to_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 处理可能的 float 类型图片
        if img.dtype != np.uint8:
            if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
            else: img = img.astype(np.uint8)
            
        ret, buf = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)],
        )
        return base64.b64encode(buf).decode("utf-8") if ret else ""

    def _blank_frame(self, w: int, h: int, text: str) -> np.ndarray:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        return frame

    def _get_bgr_frame(self, raw_obs: Dict[str, Any], obs_key: str, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        img = raw_obs.get(obs_key)
        if not isinstance(img, np.ndarray):
            return self._blank_frame(w, h, f"missing: {obs_key}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        return img

    def _render_curve(self, values: list[float], title: str, size: tuple[int, int], color_bgr: tuple[int, int, int]) -> np.ndarray:
        w, h = size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        if len(values) < 2:
            return canvas

        plot_x0, plot_y0 = 60, 50
        plot_x1, plot_y1 = w - 20, h - 40
        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (80, 80, 80), 1)

        v = np.asarray(values, dtype=np.float32)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        pad = 0.05 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        n = len(v)
        xs = np.linspace(plot_x0, plot_x1, n, dtype=np.float32)
        ys = plot_y1 - (v - vmin) / (vmax - vmin) * (plot_y1 - plot_y0)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color_bgr, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"{v[-1]:.4f}", (plot_x1 - 160, plot_y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)
        return canvas

    def _finalize_recording(self, label: str) -> None:
        os.makedirs(self.config.result_dir, exist_ok=True)
        label_dir = os.path.join(self.config.result_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        dst = os.path.join(label_dir, self._run_time)
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(self._staging_dir, dst)
            print(f"✅ 已保存到: {dst}")
        except Exception as e:
            print(f"⚠️ 保存失败: {e} (staging_dir={self._staging_dir}, dst={dst})")

    def _prepare_observation(self, raw_obs):
        # IMPORTANT: state + images must come from the SAME raw_obs frame.
        state = self._extract_state(raw_obs)
        images = {}
        for obs_key, train_name in self.camera_mapping.items():
            if obs_key in raw_obs:
                images[train_name] = self._encode_image(raw_obs[obs_key])
        return {"state": state, "images": images}

    def control_loop(self):
        try:
            while not self._is_finished:
                loop_start = time.perf_counter()
                raw_obs = self.robot.get_observation()  # single read per step
                obs_req = self._prepare_observation(raw_obs)

                self.ws.send(json.dumps(obs_req, ensure_ascii=False))
                resp = self.ws.recv(timeout=self.config.ws_timeout_s)
                result = json.loads(resp)
                
                if "error" in result:
                    print(f"❌ 服务器推理失败: {result['error']}")
                    continue

                action_list = result.get("action")
                if not isinstance(action_list, list):
                    raise ValueError(f"Server response missing 'action' list: keys={list(result.keys())}")
                if len(action_list) != 14:
                    raise ValueError(f"Expected 14-d action (right7+left7), got len={len(action_list)}")

                # Server outputs are interpreted as right7 + left7 absolute targets.
                raw_action = [float(x) for x in action_list]
                if self.config.log_server_action_every_step:
                    formatted = ", ".join(f"{x:.4f}" for x in raw_action)
                    print(f"[ServerAction step={self.step_idx}] [{formatted}]")

                right = raw_action[:7]
                left = raw_action[7:]

                if self.freeze_left_arm:
                    # Keep left arm at current observation frame pose (same frame as images/state).
                    left = [float(self._to_native(raw_obs.get(k, 0.0))) for k in self._left_state_keys]

                right_gripper_raw = right[6]
                left_gripper_raw = left[6]

                # 将训练数据中的夹爪开合尺度映射到机器人命令尺度(米)
                right[6] = self._map_gripper_action(right[6])
                left[6] = self._map_gripper_action(left[6])

                action_cmd = {
                    "left_joint_1_pos": left[0], "left_joint_2_pos": left[1],
                    "left_joint_3_pos": left[2], "left_joint_4_pos": left[3],
                    "left_joint_5_pos": left[4], "left_joint_6_pos": left[5],
                    "left_gripper_pos": left[6],
                    "right_joint_1_pos": right[0], "right_joint_2_pos": right[1],
                    "right_joint_3_pos": right[2], "right_joint_4_pos": right[3],
                    "right_joint_5_pos": right[4], "right_joint_6_pos": right[5],
                    "right_gripper_pos": right[6],
                }
                self.robot.send_action(action_cmd)

                self._curve_left_grip_raw.append(float(left_gripper_raw))
                self._curve_left_grip_cmd.append(float(left[6]))
                self._curve_right_grip_raw.append(float(right_gripper_raw))
                self._curve_right_grip_cmd.append(float(right[6]))

                cam_size = (640, 480)
                left_bgr = self._get_bgr_frame(raw_obs, "observation.images.image_left", cam_size)
                right_bgr = self._get_bgr_frame(raw_obs, "observation.images.image_right", cam_size)
                top_bgr = self._get_bgr_frame(raw_obs, "observation.images.image_top", cam_size)

                camera_strip = np.concatenate([left_bgr, top_bgr, right_bgr], axis=1)
                self._writers.write("cameras_left_mid_right", camera_strip)

                plot_size = (640, 480)
                curve1 = self._render_curve(self._curve_left_grip_raw, "Left Gripper Raw", plot_size, (0, 200, 0))
                curve2 = self._render_curve(self._curve_left_grip_cmd, "Left Gripper Cmd(m)", plot_size, (0, 200, 200))
                curve3 = self._render_curve(self._curve_right_grip_raw, "Right Gripper Raw", plot_size, (200, 0, 200))
                curve4 = self._render_curve(self._curve_right_grip_cmd, "Right Gripper Cmd(m)", plot_size, (200, 200, 0))
                self._writers.write("curve_left_gripper_raw", curve1)
                self._writers.write("curve_left_gripper_cmd", curve2)
                self._writers.write("curve_right_gripper_raw", curve3)
                self._writers.write("curve_right_gripper_cmd", curve4)

                if self.config.show_window:
                    try:
                        dashboard = np.zeros((960, 1920, 3), dtype=np.uint8)
                        dashboard[0:480, 0:1920] = camera_strip
                        p1 = cv2.resize(curve1, (960, 240))
                        p2 = cv2.resize(curve2, (960, 240))
                        p3 = cv2.resize(curve3, (960, 240))
                        p4 = cv2.resize(curve4, (960, 240))
                        dashboard[480:720, 0:960] = p1
                        dashboard[480:720, 960:1920] = p2
                        dashboard[720:960, 0:960] = p3
                        dashboard[720:960, 960:1920] = p4
                        cv2.putText(
                            dashboard,
                            "Window: Press c=success, f=fail | Terminal: Type c or f",
                            (20, 470),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(self._window_name, dashboard)
                        key = cv2.waitKeyEx(10)
                        if key != -1 and key != 255:
                            print(f"Key pressed: {key}")
                        key_chr = (key & 0xFF) if key not in (-1, 255) else -1
                        if key_chr in (ord("c"), ord("C")):
                            self._label = "success"
                            self._is_finished = True
                        elif key_chr in (ord("f"), ord("F")):
                            self._label = "fail"
                            self._is_finished = True
                        elif key_chr == 27:
                            self._label = "unkown"
                            self._is_finished = True
                    except cv2.error as e:
                        print(f"⚠️ OpenCV 无 GUI 支持，已自动关闭窗口显示（仍会录制视频）。错误: {e}")
                        self.config.show_window = False
                
                if self.step_idx % 30 == 0:
                    print(
                        f"Step {self.step_idx} | "
                        f"L_J1: {left[0]:.3f} | R_J1: {right[0]:.3f} | "
                        f"L_Grip raw->cmd: {left_gripper_raw:.4f}->{left[6]:.4f} | "
                        f"R_Grip raw->cmd: {right_gripper_raw:.4f}->{right[6]:.4f}"
                    )

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(1.0 / self.frequency - elapsed, 0.0))
                self.step_idx += 1
                
        except KeyboardInterrupt:
            print("\n用户中断控制循环")
            if self._label is None:
                self._label = "unkown"
        except TimeoutError:
            print(f"❌ WebSocket recv 超时(>{self.config.ws_timeout_s}s)，请检查服务端是否卡住/网络是否稳定")
            if self._label is None:
                self._label = "unkown"
        except Exception as e:
            print(f"❌ 严重崩溃: {e}")
            traceback.print_exc()
            if self._label is None:
                self._label = "unkown"
        finally:
            try:
                self._writers.close()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass
            try:
                self.ws.close()
            except Exception:
                pass
            try:
                self._finalize_recording(self._label or "unkown")
            except Exception:
                pass
            print("硬件资源和 Socket 已释放")

@draccus.wrap()
def main(cfg: ACTClientConfig):
    client = ACTClient(cfg)
    client.control_loop()

if __name__ == "__main__":
    main()
