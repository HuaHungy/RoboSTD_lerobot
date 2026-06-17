"""
python src/openpi_client.py \
  --host 192.168.10.1 \
  --prompt "Agilex_Cobot_Magic_Put_the_bowl_on_the_plate_left" \
  --cam-high 10 \
  --cam-left-wrist 4 \
  --cam-right-wrist 16 \
  --control-hz 30 \
  --velocity 60 \
  --gripper-mode meters \
  --max-steps -1

  #man wen
python src/openpi_client.py \
  --host 192.168.10.1 \
  --prompt "Agilex_Cobot_collect_cups_left" \
  --cam-high 10 --cam-left-wrist 4 --cam-right-wrist 16 \
  --control-hz 30 \
  --velocity 30 \
  --gripper-mode dataset_076 \
  --dataset-gripper-max 0.76 \
  --max-steps -1

python src/openpi_client.py \
  --host 192.168.10.1 \
  --prompt "Agilex_Cobot_put_bread" \
  --cam-high 10 --cam-left-wrist 4 --cam-right-wrist 16 \
  --control-hz 30 \
  --velocity 30 \
  --gripper-mode dataset_076 \
  --dataset-gripper-max 0.76 \
  --max-steps -1
"""


import abc
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

from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.configuration_bi_piper import BiPiperConfig
import msgpack
import numpy as np
import tyro
import websockets.sync.client

_IMAGE_H = 224
_IMAGE_W = 224

_STDIN_QUEUE: queue.Queue = queue.Queue()


def _stdin_reader():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped:
                _STDIN_QUEUE.put(stripped)
        except (EOFError, OSError):
            break


# -- msgpack + numpy serialization --------------------------------------------------------------

def _resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by padding with black."""
    cur_height, cur_width = image.shape[:2]
    if cur_width == target_width and cur_height == target_height:
        return image

    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    # Resize the image using OpenCV
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_h0, remainder_h = divmod(target_height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(target_width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    
    # Pad the image with black (0)
    padded_image = np.pad(
        resized_image,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode='constant',
        constant_values=0
    )
    return padded_image

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


# -- Camera capture -----------------------------------------------------------------------------

class CameraCapture:
    """Reads frames from a USB camera via OpenCV, falling back to black images."""

    def __init__(self, device_idx: int):
        self._device_idx = device_idx
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
            return self._black.copy()

        ret, frame = self._cap.read()
        if not ret:
            return self._black.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_with_pad(frame, _IMAGE_H, _IMAGE_W)
        return frame  # noqa: RET504

    @property
    def is_active(self) -> bool:
        return self._cap is not None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# -- WebSocket policy client ---------------------------------------------------------------------

class _BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        ...

    def reset(self) -> None:  # noqa: B027
        pass


class WebsocketClientPolicy(_BasePolicy):
    def __init__(self, host="0.0.0.0", port=8000, recv_timeout_s: float | None = None):
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = _msgpack_packer()
        self._recv_timeout_s = recv_timeout_s
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
            logging.info("Sending first observation: state=%s images=%s prompt=%r",
                         state_shape, img_shapes, obs.get("prompt", ""))

        data = self._packer.pack(obs)

        for attempt in range(3):
            try:
                self._ws.send(data)
                response = self._ws.recv(timeout=self._recv_timeout_s)

                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")

                result = _msgpack_unpackb(response)

                self._step_count += 1
                if self._step_count == 1:
                    state_shape = np.asarray(obs["state"]).shape
                    img_shapes = {k: v.shape for k, v in obs.get("images", {}).items()}
                    logging.info("First inference succeeded. state=%s images=%s", state_shape, img_shapes)

                return result

            except websockets.exceptions.ConnectionClosed:
                logging.warning("Server disconnected (attempt %d/3).", attempt + 1)
                if attempt < 2:
                    self._reconnect()
                else:
                    raise RuntimeError(
                        "Server closed the connection during inference after 3 attempts. "
                        "The model may have crashed on the server — check the server logs. "
                        "Tip: try 'XLA_FLAGS=\"--xla_gpu_autotune_level=0\" python scripts/serve_policy.py ...'"
                    ) from None
            except TimeoutError:
                logging.warning("Timed out waiting for server response (attempt %d/3).", attempt + 1)
                if attempt < 2:
                    self._reconnect()
                else:
                    raise RuntimeError(
                        f"Timed out waiting for inference response after 3 attempts (timeout={self._recv_timeout_s}s)."
                    ) from None

        raise AssertionError("unreachable")

    def reset(self) -> None:
        pass


# -- Action chunk broker ------------------------------------------------------------------------

class ActionChunkBroker(_BasePolicy):
    def __init__(self, policy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._last_results = None

    def infer(self, obs: dict) -> dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        out = {}
        for k, v in self._last_results.items():
            out[k] = v[self._cur_step, ...] if isinstance(v, np.ndarray) else v

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None
        return out

    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


# -- Main entry ---------------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    server_recv_timeout_s: float | None = 60.0

    can_left: str = "can_left"
    can_right: str = "can_right"
    velocity: int = 40

    cam_high: int = -1
    cam_low: int = -1
    cam_left_wrist: int = -1
    cam_right_wrist: int = -1

    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.07
    gripper_invert: bool = True
    gripper_mode: Literal["aloha", "meters", "dataset_076"] = "aloha"
    dataset_gripper_max: float = 0.76
    log_action_debug: bool = True
    action_mode: Literal["auto", "absolute", "delta_joints"] = "auto"
    joint_step_scale: float = 1.0
    max_joint_delta_rad: float = 0.15
    send_repeats: int = 2

    max_steps: int = 300
    action_horizon: int = 10
    control_hz: int = 20
    prompt: str = "do something"


def _gripper_to_aloha(gripper_m: float, min_m: float, max_m: float, invert: bool) -> float:
    clamped = float(np.clip(gripper_m, min_m, max_m))
    normalized = (clamped - min_m) / (max_m - min_m)
    return 1.0 - normalized if invert else normalized


def _gripper_from_aloha(gripper_aloha: float, min_m: float, max_m: float, invert: bool) -> float:
    clamped = float(np.clip(gripper_aloha, 0.0, 1.0))
    normalized = 1.0 - clamped if invert else clamped
    return min_m + normalized * (max_m - min_m)


def main(args: Args) -> None:
    logging.info("Connecting to policy server at %s:%d ...", args.host, args.port)
    policy = WebsocketClientPolicy(host=args.host, port=args.port, recv_timeout_s=args.server_recv_timeout_s)
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
    if all(s == "black" for s in cam_status.values()):
        logging.warning(
            "ALL cameras are using black placeholders! "
            "Pi0 is a vision-language-action model — it needs real camera input to understand "
            "the scene and decide what actions to take. With black images the model will output "
            "near-zero actions and the robot will stay near its initial position. "
            "Use --cam-high <N> --cam-low <N> --cam-left-wrist <N> --cam-right-wrist <N> "
            "to specify USB camera device indices (try 'ls /dev/video*' on the robot machine)."
        )

    config = BiPiperConfig(can_left=args.can_left, can_right=args.can_right, velocity=args.velocity)
    robot = BiPiper(config)
    robot.connect()
    logging.info("BiPiper robot connected.")

    threading.Thread(target=_stdin_reader, daemon=True).start()
    logging.info("Stdin reader started. Type a string and press Enter to save episode video.")

    episode_name: str | None = None
    video_frames: list = []

    show_window = True
    if show_window and not os.environ.get("DISPLAY"):
        show_window = False
        logging.warning("No DISPLAY detected, disabling window.")
    window_name = "OpenPI Client"
    if show_window and _HAS_CV2:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
        except cv2.error as e:
            show_window = False
            logging.warning("OpenCV has no GUI support, disabling window. Error: %s", e)

    step = 0
    try:
        while args.max_steps < 0 or step < args.max_steps:
            loop_start = time.time()

            obs = robot.get_observation()

            left_state = [obs[f"left_{name}_pos"] for name in config.joint_names]
            right_state = [obs[f"right_{name}_pos"] for name in config.joint_names]
            state = np.array(right_state + left_state, dtype=np.float64)

            state_gripper_right_raw = float(state[6])
            state_gripper_left_raw = float(state[13])
            if args.gripper_mode == "aloha":
                state[6] = _gripper_to_aloha(
                    state[6], args.gripper_min_m, args.gripper_max_m, args.gripper_invert
                )
                state[13] = _gripper_to_aloha(
                    state[13], args.gripper_min_m, args.gripper_max_m, args.gripper_invert
                )
            elif args.gripper_mode == "dataset_076":
                denom = max(args.gripper_max_m - args.gripper_min_m, 1e-9)
                norm_r = float(np.clip((state[6] - args.gripper_min_m) / denom, 0.0, 1.0))
                norm_l = float(np.clip((state[13] - args.gripper_min_m) / denom, 0.0, 1.0))
                state[6] = norm_r * float(args.dataset_gripper_max)
                state[13] = norm_l * float(args.dataset_gripper_max)
            elif args.gripper_mode == "meters":
                pass
            else:
                raise ValueError(f"Unknown gripper_mode: {args.gripper_mode}")

            images_for_model = {}
            display_images = []
            raw_images = {}
            for name, cam in cameras.items():
                img_rgb = cam.read()
                raw_images[name] = img_rgb
                images_for_model[name] = img_rgb.transpose(2, 0, 1).copy()
                if show_window and _HAS_CV2:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    display_images.append(img_bgr)

            stitched_rgb = np.concatenate(
                (
                    raw_images["cam_left_wrist"],
                    raw_images["cam_high"],
                    raw_images["cam_right_wrist"],
                ),
                axis=1,
            )
            stitched_bgr = cv2.cvtColor(stitched_rgb, cv2.COLOR_RGB2BGR)
            video_frames.append(stitched_bgr)

            try:
                while True:
                    user_input = _STDIN_QUEUE.get_nowait()
                    if user_input:
                        episode_name = user_input
                        logging.info("Episode name set to: %s, stopping episode. %d frames recorded.", episode_name, len(video_frames))
                        break
            except queue.Empty:
                pass
            else:
                break

            if show_window and _HAS_CV2 and display_images:
                try:
                    if len(display_images) == 4:
                        top_row = np.concatenate(display_images[:2], axis=1)
                        bottom_row = np.concatenate(display_images[2:], axis=1)
                        dashboard = np.concatenate([top_row, bottom_row], axis=0)
                    else:
                        dashboard = np.concatenate(display_images, axis=1)
                    
                    dashboard = cv2.resize(dashboard, (dashboard.shape[1] * 2, dashboard.shape[0] * 2))
                    
                    cv2.imshow(window_name, dashboard)
                    key = cv2.waitKeyEx(10)
                    if key != -1 and key != 255:
                        key_chr = (key & 0xFF)
                        if key_chr == 27:  # ESC
                            logging.info("ESC pressed, stopping.")
                            break
                except cv2.error as e:
                    logging.warning("OpenCV imshow error: %s", e)
                    show_window = False

            observation = {
                "state": state,
                "images": images_for_model,
                "prompt": args.prompt,
            }

            action = broker.infer(observation)
            # msgpack can return numpy arrays backed by read-only buffers; ensure we can edit in-place.
            action_vec = np.array(action["actions"], dtype=np.float64, copy=True)
            server_action_raw = action_vec.copy()

            joint_idxs = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            action_vec[joint_idxs] = np.clip(action_vec[joint_idxs], -3.14, 3.14)
            action_gripper_right_raw = float(server_action_raw[6])
            action_gripper_left_raw = float(server_action_raw[13])
            joint_diff = server_action_raw[joint_idxs] - state[joint_idxs]
            mean_abs_diff = float(np.mean(np.abs(joint_diff)))
            max_abs_diff = float(np.max(np.abs(joint_diff)))
            mean_abs_action = float(np.mean(np.abs(server_action_raw[joint_idxs])))
            mean_abs_state = float(np.mean(np.abs(state[joint_idxs])))
            likely_delta = mean_abs_action < 0.2 and mean_abs_state > 0.5

            if args.action_mode == "delta_joints" or (args.action_mode == "auto" and likely_delta):
                action_vec[joint_idxs] = action_vec[joint_idxs] + state[joint_idxs]

            if args.joint_step_scale != 1.0:
                action_vec[joint_idxs] = state[joint_idxs] + (action_vec[joint_idxs] - state[joint_idxs]) * float(
                    args.joint_step_scale
                )

            if args.max_joint_delta_rad is not None and args.max_joint_delta_rad > 0:
                max_delta = float(args.max_joint_delta_rad)
                action_vec[joint_idxs] = state[joint_idxs] + np.clip(
                    action_vec[joint_idxs] - state[joint_idxs], -max_delta, max_delta
                )

            if args.gripper_mode == "aloha":
                action_vec[6] = _gripper_from_aloha(
                    action_vec[6], args.gripper_min_m, args.gripper_max_m, args.gripper_invert
                )
                action_vec[13] = _gripper_from_aloha(
                    action_vec[13], args.gripper_min_m, args.gripper_max_m, args.gripper_invert
                )
            elif args.gripper_mode == "dataset_076":
                denom = max(float(args.dataset_gripper_max), 1e-9)
                norm_r = float(np.clip(action_vec[6] / denom, 0.0, 1.0))
                norm_l = float(np.clip(action_vec[13] / denom, 0.0, 1.0))
                action_vec[6] = args.gripper_min_m + norm_r * (args.gripper_max_m - args.gripper_min_m)
                action_vec[13] = args.gripper_min_m + norm_l * (args.gripper_max_m - args.gripper_min_m)
            elif args.gripper_mode == "meters":
                pass
            else:
                raise ValueError(f"Unknown gripper_mode: {args.gripper_mode}")

            action_msg = {}
            for i, name in enumerate(config.joint_names):
                action_msg[f"right_{name}_pos"] = float(action_vec[i])
                action_msg[f"left_{name}_pos"] = float(action_vec[7 + i])

            if args.log_action_debug:
                mode = args.action_mode
                if mode == "auto":
                    mode = "delta_joints" if likely_delta else "absolute"
                logging.info(
                    "[step %d] server_action(joints) vs state: mean|a-state|=%.4f max|a-state|=%.4f "
                    "mean|a|=%.4f mean|state|=%.4f likely_delta=%s | gripper_raw=(R%.4f,L%.4f) mode=%s",
                    step,
                    mean_abs_diff,
                    max_abs_diff,
                    mean_abs_action,
                    mean_abs_state,
                    likely_delta,
                    float(server_action_raw[6]),
                    float(server_action_raw[13]),
                    f"{mode}/{args.gripper_mode}",
                )
                logging.info(
                    "[step %d] gripper: state_raw(m)=(R%.4f,L%.4f) state_sent=(R%.4f,L%.4f) "
                    "action_raw=(R%.4f,L%.4f) action_cmd(m)=(R%.4f,L%.4f) invert=%s",
                    step,
                    state_gripper_right_raw,
                    state_gripper_left_raw,
                    float(state[6]),
                    float(state[13]),
                    action_gripper_right_raw,
                    action_gripper_left_raw,
                    float(action_vec[6]),
                    float(action_vec[13]),
                    str(args.gripper_invert),
                )

            right_vals = [f"{action_vec[i]:.3f}" for i in range(7)]
            left_vals = [f"{action_vec[7 + i]:.3f}" for i in range(7)]
            logging.info(
                "[step %d] R: %s | L: %s",
                step, " ".join(right_vals), " ".join(left_vals),
            )

            repeats = int(args.send_repeats) if args.send_repeats is not None else 1
            if repeats < 1:
                repeats = 1
            for _ in range(repeats):
                robot.send_action(action_msg)

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 1.0 / args.control_hz - elapsed)
            time.sleep(sleep_time)
            step += 1

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    else:
        if args.max_steps >= 0 and step >= args.max_steps:
            episode_name = "f"
            logging.info("Reached max_steps=%d, stopping.", args.max_steps)
    finally:
        if not episode_name and video_frames:
            episode_name = "unknown"
            logging.info("Episode name not set, saving to 'unknown' directory.")
        if episode_name and video_frames:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result_dir = os.path.join(os.path.dirname(script_dir), "result", episode_name)
            os.makedirs(result_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(result_dir, f"{timestamp}.mp4")
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, float(args.control_hz), (w, h))
            for frame in video_frames:
                writer.write(frame)
            writer.release()
            logging.info("Episode video saved to: %s (%d frames)", output_path, len(video_frames))

        for cam in cameras.values():
            cam.close()
        robot.disconnect()
        if show_window and _HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logging.info("Robot disconnected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
