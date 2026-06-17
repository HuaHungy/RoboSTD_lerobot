"""
python src/openpi_client_fast.py \
  --host 192.168.10.1 \
  --prompt "Agilex_Cobot_put_bread" \
  --cam-high 10 --cam-left-wrist 4 --cam-right-wrist 16 \
  --control-hz 30 \
  --velocity 3 \
  --gripper-mode dataset_076 \
  --dataset-gripper-max 0.76 \
  --max-steps -1
"""

import abc
import concurrent.futures
import dataclasses
import functools
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Literal

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

import msgpack
import numpy as np
import tyro
import websockets.sync.client

from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.configuration_bi_piper import BiPiperConfig

_IMAGE_H = 224
_IMAGE_W = 224
_JOINT_IDXS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
_GRIPPER_IDXS = np.array([6, 13], dtype=np.int64)
_STDIN_QUEUE: queue.Queue[str] = queue.Queue()


def _stdin_reader() -> None:
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, OSError):
            break
        if not line:
            break
        stripped = line.strip()
        if stripped:
            _STDIN_QUEUE.put(stripped)


def _drain_stdin_queue() -> str | None:
    latest = None
    while True:
        try:
            latest = _STDIN_QUEUE.get_nowait()
        except queue.Empty:
            return latest


def _resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    cur_height, cur_width = image.shape[:2]
    if cur_width == target_width and cur_height == target_height:
        return image

    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_h0, remainder_h = divmod(target_height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(target_width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    return np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _msgpack_pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _msgpack_unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_msgpack_packer = functools.partial(msgpack.Packer, default=_msgpack_pack_array)
_msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_msgpack_unpack_array)


class CameraCapture:
    def __init__(self, device_idx: int):
        self._cap = None
        self._black = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)

        if device_idx < 0:
            logging.info("Camera device index is %d, using black placeholder.", device_idx)
            return
        if not _HAS_CV2:
            logging.warning("cv2 not installed, using black placeholder for camera %d.", device_idx)
            return

        self._cap = cv2.VideoCapture(device_idx)
        if not self._cap.isOpened():
            logging.warning("Cannot open camera %d, using black placeholder.", device_idx)
            self._cap.release()
            self._cap = None

    def read(self) -> np.ndarray:
        if self._cap is None:
            return self._black
        ret, frame = self._cap.read()
        if not ret:
            return self._black
        return _resize_with_pad(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), _IMAGE_H, _IMAGE_W)

    @property
    def is_active(self) -> bool:
        return self._cap is not None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class _BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        ...

    def reset(self) -> None:
        return None


class WebsocketClientPolicy(_BasePolicy):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, recv_timeout_s: float | None = None):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._recv_timeout_s = recv_timeout_s
        self._packer = _msgpack_packer()
        self._ws, self._server_metadata = self._wait_for_server()
        self._step_count = 0

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = _msgpack_unpackb(conn.recv(timeout=self._recv_timeout_s))
                return conn, metadata
            except OSError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _reconnect(self) -> None:
        logging.info("Reconnecting to server...")
        self._ws, self._server_metadata = self._wait_for_server()

    def infer(self, obs: dict) -> dict:
        if self._step_count == 0:
            state_shape = np.asarray(obs["state"]).shape
            img_shapes = {k: v.shape for k, v in obs.get("images", {}).items()}
            logging.info("Sending first observation: state=%s images=%s prompt=%r", state_shape, img_shapes, obs["prompt"])

        payload = self._packer.pack(obs)
        for attempt in range(3):
            try:
                self._ws.send(payload)
                response = self._ws.recv(timeout=self._recv_timeout_s)
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                result = _msgpack_unpackb(response)
                self._step_count += 1
                if self._step_count == 1:
                    logging.info("First inference succeeded.")
                return result
            except websockets.exceptions.ConnectionClosed:
                logging.warning("Server disconnected (attempt %d/3).", attempt + 1)
                if attempt < 2:
                    self._reconnect()
                    continue
                raise RuntimeError("Server closed the connection during inference after 3 attempts.") from None
            except TimeoutError:
                logging.warning("Timed out waiting for server response (attempt %d/3).", attempt + 1)
                if attempt < 2:
                    self._reconnect()
                    continue
                raise RuntimeError(
                    f"Timed out waiting for inference response after 3 attempts (timeout={self._recv_timeout_s}s)."
                ) from None
        raise AssertionError("unreachable")


class ActionChunkBroker(_BasePolicy):
    def __init__(self, policy: _BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = max(1, int(action_horizon))
        self._cur_step = 0
        self._last_results = None

    def _result_horizon(self) -> int | None:
        if self._last_results is None:
            return None
        horizons = []
        for value in self._last_results.values():
            if isinstance(value, np.ndarray):
                if value.ndim >= 2:
                    horizons.append(int(value.shape[0]))
                elif value.ndim == 1:
                    horizons.append(1)
        return min(horizons) if horizons else None

    def infer(self, obs: dict) -> dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        available = self._result_horizon()
        if available is not None and self._cur_step >= min(self._action_horizon, available):
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0
            available = self._result_horizon()

        out = {}
        for key, value in self._last_results.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                out[key] = value[self._cur_step, ...]
            else:
                out[key] = value

        self._cur_step += 1
        if available is None:
            if self._cur_step >= self._action_horizon:
                self._last_results = None
        elif self._cur_step >= min(self._action_horizon, available):
            self._last_results = None
        return out

    def reset(self) -> None:
        self._policy.reset()
        self._cur_step = 0
        self._last_results = None


class ActionSmoother:
    def __init__(self, joint_alpha: float, gripper_alpha: float):
        self._joint_alpha = float(np.clip(joint_alpha, 0.0, 1.0))
        self._gripper_alpha = float(np.clip(gripper_alpha, 0.0, 1.0))
        self._last_action: np.ndarray | None = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self._last_action is None:
            self._last_action = action.copy()
            return action

        out = action.copy()
        out[_JOINT_IDXS] = (
            self._joint_alpha * out[_JOINT_IDXS] + (1.0 - self._joint_alpha) * self._last_action[_JOINT_IDXS]
        )
        out[_GRIPPER_IDXS] = (
            self._gripper_alpha * out[_GRIPPER_IDXS] + (1.0 - self._gripper_alpha) * self._last_action[_GRIPPER_IDXS]
        )
        self._last_action = out.copy()
        return out


class StreamVideoRecorder:
    def __init__(self, enabled: bool, fps: int, result_root: str):
        self._enabled = enabled and _HAS_CV2
        self._fps = float(fps)
        self._result_root = result_root
        self._writer = None
        self._tmp_path: str | None = None
        self._frame_count = 0
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, frame_bgr: np.ndarray) -> None:
        if not self._enabled:
            return
        if self._writer is None:
            tmp_dir = os.path.join(self._result_root, "_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self._tmp_path = os.path.join(tmp_dir, f"{self._timestamp}_{os.getpid()}.mp4")
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._tmp_path, fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                logging.warning("Failed to create video writer at %s.", self._tmp_path)
                self._writer = None
                self._enabled = False
                return
        self._writer.write(frame_bgr)
        self._frame_count += 1

    def finalize(self, episode_name: str | None) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        target_episode = episode_name or "unknown"
        target_dir = os.path.join(self._result_root, target_episode)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{self._timestamp}.mp4")
        assert self._tmp_path is not None
        os.replace(self._tmp_path, target_path)
        logging.info("Episode video saved to: %s (%d frames)", target_path, self._frame_count)


class FastBiPiperCommander:
    def __init__(self, robot: BiPiper, send_repeats: int):
        self._robot = robot
        self._send_repeats = max(1, int(send_repeats))
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="bipiper_send")

    def _send_arm(self, arm, action_model: np.ndarray) -> None:
        action_standard = arm.model_joint_transform.input_transform(action_model)
        for _ in range(self._send_repeats):
            arm.set_joint_state(action_standard)

    def send(self, action: np.ndarray) -> None:
        right = np.array(action[:7], dtype=np.float64, copy=True)
        left = np.array(action[7:], dtype=np.float64, copy=True)
        future_left = self._pool.submit(self._send_arm, self._robot.left_robot, left)
        future_right = self._pool.submit(self._send_arm, self._robot.right_robot, right)
        future_left.result()
        future_right.result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    server_recv_timeout_s: float | None = 60.0

    can_left: str = "can_left"
    can_right: str = "can_right"
    velocity: int = 30

    cam_high: int = -1
    cam_low: int = -1
    cam_left_wrist: int = -1
    cam_right_wrist: int = -1

    prompt: str = "do something"
    control_hz: int = 30
    max_steps: int = -1
    action_horizon: int = 8

    gripper_mode: Literal["dataset_076"] = "dataset_076"
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.07
    dataset_gripper_max: float = 0.76

    joint_alpha: float = 0.65
    gripper_alpha: float = 0.5
    max_joint_delta_rad: float = 0.25
    send_repeats: int = 1

    log_interval: int = 30
    log_action_debug: bool = False
    show_window: bool = True
    record_video: bool = True


def _gripper_to_dataset_scale(gripper_m: float, min_m: float, max_m: float, max_dataset: float) -> float:
    denom = max(max_m - min_m, 1e-9)
    normalized = float(np.clip((gripper_m - min_m) / denom, 0.0, 1.0))
    return normalized * float(max_dataset)


def _gripper_from_dataset_scale(gripper_value: float, min_m: float, max_m: float, max_dataset: float) -> float:
    denom = max(float(max_dataset), 1e-9)
    normalized = float(np.clip(gripper_value / denom, 0.0, 1.0))
    return min_m + normalized * (max_m - min_m)


def _annotate(name: str, image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(image_bgr, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return image_bgr


def _build_dashboard(cam_high: np.ndarray, cam_low: np.ndarray, cam_left: np.ndarray, cam_right: np.ndarray) -> np.ndarray:
    top_row = np.concatenate([_annotate("cam_high", cam_high), _annotate("cam_low", cam_low)], axis=1)
    bottom_row = np.concatenate([_annotate("cam_left_wrist", cam_left), _annotate("cam_right_wrist", cam_right)], axis=1)
    dashboard = np.concatenate([top_row, bottom_row], axis=0)
    return cv2.resize(dashboard, (dashboard.shape[1] * 2, dashboard.shape[0] * 2))


def _build_record_frame(cam_high: np.ndarray, cam_left: np.ndarray, cam_right: np.ndarray) -> np.ndarray:
    stitched_rgb = np.concatenate([cam_left, cam_high, cam_right], axis=1)
    return cv2.cvtColor(stitched_rgb, cv2.COLOR_RGB2BGR)


def main(args: Args) -> None:
    logging.info("Connecting to policy server at %s:%d ...", args.host, args.port)
    policy = WebsocketClientPolicy(args.host, args.port, args.server_recv_timeout_s)
    logging.info("Server metadata: %s", policy.get_server_metadata())
    broker = ActionChunkBroker(policy=policy, action_horizon=args.action_horizon)

    cameras = {
        "cam_high": CameraCapture(args.cam_high),
        "cam_low": CameraCapture(args.cam_low),
        "cam_left_wrist": CameraCapture(args.cam_left_wrist),
        "cam_right_wrist": CameraCapture(args.cam_right_wrist),
    }
    cam_status = {name: "ok" if cam.is_active else "black" for name, cam in cameras.items()}
    logging.info("Camera status: %s", cam_status)

    robot = BiPiper(BiPiperConfig(can_left=args.can_left, can_right=args.can_right, velocity=args.velocity))
    robot.connect()
    logging.info("BiPiper robot connected.")

    sender = FastBiPiperCommander(robot, send_repeats=args.send_repeats)
    smoother = ActionSmoother(joint_alpha=args.joint_alpha, gripper_alpha=args.gripper_alpha)

    threading.Thread(target=_stdin_reader, daemon=True).start()
    logging.info("Stdin reader started. Type an episode name and press Enter to save video.")

    show_window = bool(args.show_window and _HAS_CV2)
    if show_window and not os.environ.get("DISPLAY"):
        show_window = False
        logging.warning("No DISPLAY detected, disabling window.")
    window_name = "OpenPI Client Fast"
    if show_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            show_window = False
            logging.warning("OpenCV has no GUI support, disabling window. Error: %s", exc)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    recorder = StreamVideoRecorder(
        enabled=args.record_video,
        fps=args.control_hz,
        result_root=os.path.join(os.path.dirname(script_dir), "result"),
    )

    episode_name: str | None = None
    step = 0
    total_start = time.perf_counter()
    try:
        while args.max_steps < 0 or step < args.max_steps:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            state = np.array(
                [obs[f"right_{name}_pos"] for name in robot.config.joint_names]
                + [obs[f"left_{name}_pos"] for name in robot.config.joint_names],
                dtype=np.float64,
            )
            state[6] = _gripper_to_dataset_scale(state[6], args.gripper_min_m, args.gripper_max_m, args.dataset_gripper_max)
            state[13] = _gripper_to_dataset_scale(
                state[13], args.gripper_min_m, args.gripper_max_m, args.dataset_gripper_max
            )

            cam_high = cameras["cam_high"].read()
            cam_left = cameras["cam_left_wrist"].read()
            cam_right = cameras["cam_right_wrist"].read()
            cam_low = cameras["cam_low"].read() if cameras["cam_low"].is_active else cam_high

            recorder.add(_build_record_frame(cam_high, cam_left, cam_right))

            typed_episode = _drain_stdin_queue()
            if typed_episode:
                episode_name = typed_episode
                logging.info("Episode name set to: %s, stopping episode.", episode_name)
                break

            if show_window:
                try:
                    cv2.imshow(window_name, _build_dashboard(cam_high, cam_low, cam_left, cam_right))
                    key = cv2.waitKeyEx(1)
                    if key != -1 and key != 255 and (key & 0xFF) == 27:
                        logging.info("ESC pressed, stopping.")
                        break
                except cv2.error as exc:
                    logging.warning("OpenCV imshow error: %s", exc)
                    show_window = False

            observation = {
                "state": state,
                "images": {
                    "cam_high": np.ascontiguousarray(np.transpose(cam_high, (2, 0, 1))),
                    "cam_low": np.ascontiguousarray(np.transpose(cam_low, (2, 0, 1))),
                    "cam_left_wrist": np.ascontiguousarray(np.transpose(cam_left, (2, 0, 1))),
                    "cam_right_wrist": np.ascontiguousarray(np.transpose(cam_right, (2, 0, 1))),
                },
                "prompt": args.prompt,
            }

            result = broker.infer(observation)
            action = np.array(result["actions"], dtype=np.float64, copy=True).reshape(-1)
            if action.shape[0] != 14:
                raise ValueError(f"Expected 14-d action, got shape={action.shape}")

            server_grippers = (float(action[6]), float(action[13]))
            action[_JOINT_IDXS] = np.clip(action[_JOINT_IDXS], -3.14, 3.14)
            action[6] = _gripper_from_dataset_scale(
                action[6], args.gripper_min_m, args.gripper_max_m, args.dataset_gripper_max
            )
            action[13] = _gripper_from_dataset_scale(
                action[13], args.gripper_min_m, args.gripper_max_m, args.dataset_gripper_max
            )

            if args.max_joint_delta_rad > 0:
                state_joint = np.array(
                    [obs[f"right_{name}_pos"] for name in robot.config.joint_names[:6]]
                    + [obs[f"left_{name}_pos"] for name in robot.config.joint_names[:6]],
                    dtype=np.float64,
                )
                target_joint = action[_JOINT_IDXS]
                delta = np.clip(target_joint - state_joint, -args.max_joint_delta_rad, args.max_joint_delta_rad)
                action[_JOINT_IDXS] = state_joint + delta

            action = smoother.apply(action)
            sender.send(action)

            if args.log_action_debug and (step % max(1, args.log_interval) == 0):
                logging.info(
                    "[step %d] gripper raw(dataset)=(R%.4f,L%.4f) cmd(m)=(R%.4f,L%.4f)",
                    step,
                    server_grippers[0],
                    server_grippers[1],
                    float(action[6]),
                    float(action[13]),
                )

            if step % max(1, args.log_interval) == 0:
                right_vals = " ".join(f"{action[i]:.3f}" for i in range(7))
                left_vals = " ".join(f"{action[7 + i]:.3f}" for i in range(7))
                elapsed_total = time.perf_counter() - total_start
                avg_fps = (step + 1) / elapsed_total if elapsed_total > 0 else 0.0
                logging.info("[step %d] fps(avg)=%.2f | R: %s | L: %s", step, avg_fps, right_vals, left_vals)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, 1.0 / args.control_hz - elapsed))
            step += 1

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        recorder.finalize(episode_name)
        sender.close()
        for cam in cameras.values():
            cam.close()
        robot.disconnect()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logging.info("Robot disconnected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
