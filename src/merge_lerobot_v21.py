#!/usr/bin/env python3

from __future__ import annotations

"""Merge multiple LeRobot v2.1 dataset folders into one multi-episode dataset."""
"""
conda activate lerobot
python '/home/agilex/.why/lerobot/src/merge_lerobot_v21.py' \
    --input-root ~/DoRobot/dataset/20260601/user/'Franka_Put the sandwich in the basket_put_vegetable_2364' \
    --output-root ./data/Agilex_Cobot_Magic_basket_put_vegetable_0601
scp -r ./data/Agilex_Cobot_Magic_basket_put_vegetable_0601/ huahungy@192.168.10.1:~/act/data/

conda activate lerobot
python '/home/agilex/.why/lerobot/src/merge_lerobot_v21.py' \
    --input-root ~/DoRobot/dataset/20260601/user/'Franka_Put the sandwich in the basket_put_tomato_2363' \
    --output-root ./data/Agilex_Cobot_Magic_basket_put_tomato_0601
scp -r ./data/Agilex_Cobot_Magic_basket_put_tomato_0601/ huahungy@192.168.10.1:~/act/data/

conda activate lerobot
python '/home/agilex/.why/lerobot/src/merge_lerobot_v21.py' \
    --input-root ~/DoRobot/dataset/Put_sandwich_in_basket  \
    --output-root ./data/Put_sandwich_in_basket_0605 
scp -r ./data/Put_sandwich_in_basket_0605/ huahungy@192.168.10.1:~/act/data/

"""


import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
)

JOINT14_STATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19]

DEFAULT_FFMPEG_BIN = Path("/opt/ffmpeg-btbn/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg")


def read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读取 JSON 失败: {path}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OSError(f"读取 JSONL 失败: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 解析失败: {path} 第 {line_number} 行") from exc
    return rows


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            delete=False,
            prefix=path.name + ".",
            suffix=".tmp",
        ) as fp:
            tmp_path = Path(fp.name)
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=4, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    atomic_write_text(path, content + ("\n" if rows else ""))


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def parse_episode_index_from_name(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d{6})", path.stem)
    if match is None:
        raise ValueError(f"无法从文件名解析 episode 编号: {path}")
    return int(match.group(1))


def format_episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def chunk_name(episode_index: int, chunk_size: int) -> str:
    return f"chunk-{episode_index // chunk_size:03d}"


def build_output_features(
    source_features: dict[str, Any],
    state_layout: str,
) -> dict[str, Any]:
    output_features = copy.deepcopy(source_features)
    if state_layout == "preserve":
        return output_features

    if state_layout != "joint14":
        raise ValueError(f"不支持的 state 布局: {state_layout}")

    if "observation.state" not in output_features:
        raise ValueError("features 中缺少 observation.state，无法输出 joint14")

    state_feature = copy.deepcopy(output_features["observation.state"])
    if not isinstance(state_feature, dict) or "names" not in state_feature:
        raise ValueError("features.observation.state 缺少 names，无法输出 joint14")
    state_feature["names"] = [state_feature["names"][index] for index in JOINT14_STATE_INDICES]
    state_feature["shape"] = [len(JOINT14_STATE_INDICES)]
    output_features["observation.state"] = state_feature
    return output_features


def coerce_features_dtype(features: dict[str, Any], keys: list[str], dtype: str) -> dict[str, Any]:
    updated = copy.deepcopy(features)
    for key in keys:
        value = updated.get(key)
        if isinstance(value, dict):
            value = copy.deepcopy(value)
            value["dtype"] = dtype
            updated[key] = value
    return updated


@dataclass
class SourceDataset:
    root: Path
    info: dict[str, Any]
    common_record: dict[str, Any] | None
    tasks_by_index: dict[int, str]
    episodes_by_index: dict[int, dict[str, Any]]
    stats_by_index: dict[int, dict[str, Any]]
    op_dataid_by_index: dict[int, dict[str, Any]]
    parquet_by_episode: dict[int, Path]
    video_keys: list[str]
    image_keys: list[str]


def discover_sources(input_root: Path, output_root: Path) -> list[Path]:
    output_resolved = output_root.resolve()
    candidates: list[Path] = []
    for child in sorted(input_root.iterdir(), key=natural_sort_key):
        if not child.is_dir():
            continue
        if child.resolve() == output_resolved:
            continue
        if all((child / relative_path).exists() for relative_path in REQUIRED_META_FILES):
            candidates.append(child)
    return candidates


def load_source_dataset(root: Path) -> SourceDataset:
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{root} 不是 LeRobot v2.1 数据集")

    if "features" not in info or not isinstance(info["features"], dict):
        raise ValueError(f"{root} 的 meta/info.json 缺少 features")
    if "chunks_size" not in info:
        raise ValueError(f"{root} 的 meta/info.json 缺少 chunks_size")
    try:
        chunk_size = int(info["chunks_size"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{root} 的 chunks_size 非法: {info.get('chunks_size')}") from exc
    if chunk_size <= 0:
        raise ValueError(f"{root} 的 chunks_size 必须为正数: {chunk_size}")

    common_record_path = root / "meta/common_record.json"
    common_record = read_json(common_record_path) if common_record_path.exists() else None

    tasks_rows = read_jsonl(root / "meta/tasks.jsonl")
    tasks_by_index: dict[int, str] = {}
    for row in tasks_rows:
        if "task_index" not in row or "task" not in row:
            raise ValueError(f"{root} 的 tasks.jsonl 缺少 task_index/task 字段")
        task_index = int(row["task_index"])
        task_name = str(row["task"])
        if task_index in tasks_by_index and tasks_by_index[task_index] != task_name:
            raise ValueError(f"{root} 的 tasks.jsonl 存在重复 task_index={task_index}")
        tasks_by_index[task_index] = task_name

    episodes_rows = read_jsonl(root / "meta/episodes.jsonl")
    episodes_by_index: dict[int, dict[str, Any]] = {}
    for row in episodes_rows:
        if "episode_index" not in row:
            raise ValueError(f"{root} 的 episodes.jsonl 缺少 episode_index 字段")
        episode_index = int(row["episode_index"])
        if episode_index in episodes_by_index:
            raise ValueError(f"{root} 的 episodes.jsonl 存在重复 episode_index={episode_index}")
        episodes_by_index[episode_index] = row

    stats_rows = read_jsonl(root / "meta/episodes_stats.jsonl")
    stats_by_index: dict[int, dict[str, Any]] = {}
    for row in stats_rows:
        if "episode_index" not in row:
            continue
        stats_by_index[int(row["episode_index"])] = row

    op_dataid_rows = read_jsonl(root / "meta/op_dataid.jsonl")
    op_dataid_by_index: dict[int, dict[str, Any]] = {}
    for row in op_dataid_rows:
        if "episode_index" not in row:
            continue
        op_dataid_by_index[int(row["episode_index"])] = row

    parquet_by_episode: dict[int, Path] = {}
    for parquet_path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        episode_index = parse_episode_index_from_name(parquet_path)
        if episode_index in parquet_by_episode:
            raise ValueError(f"{root} 存在重复 episode parquet: episode_index={episode_index}")
        parquet_by_episode[episode_index] = parquet_path

    if not parquet_by_episode:
        raise ValueError(f"{root} 缺少 parquet 数据文件")

    if set(parquet_by_episode) != set(episodes_by_index):
        raise ValueError(
            f"{root} 的 parquet 文件与 episodes.jsonl 不一致: "
            f"parquet={sorted(parquet_by_episode)} episodes={sorted(episodes_by_index)}"
        )

    video_keys = sorted(
        key
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    )

    image_keys = sorted(
        key
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "image"
    )

    return SourceDataset(
        root=root,
        info=info,
        common_record=common_record,
        tasks_by_index=tasks_by_index,
        episodes_by_index=episodes_by_index,
        stats_by_index=stats_by_index,
        op_dataid_by_index=op_dataid_by_index,
        parquet_by_episode=parquet_by_episode,
        video_keys=video_keys,
        image_keys=image_keys,
    )


def source_signature(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "codebase_version": info.get("codebase_version"),
        "dorobot_dataset_version": info.get("dorobot_dataset_version"),
        "robot_type": info.get("robot_type"),
        "chunks_size": info.get("chunks_size"),
        "fps": info.get("fps"),
        "data_path": info.get("data_path"),
        "image_path": info.get("image_path"),
        "video_path": info.get("video_path"),
        "audio_path": info.get("audio_path"),
        "features": info.get("features"),
    }


def validate_sources(sources: list[SourceDataset]) -> None:
    if not sources:
        raise ValueError("没有发现可合并的数据集目录")

    expected_signature = source_signature(sources[0].info)
    first_common = sources[0].common_record

    for source in sources:
        if source_signature(source.info) != expected_signature:
            raise ValueError(f"{source.root} 的格式签名与其它数据集不一致，已终止合并")

        if source.common_record is None:
            if first_common is not None:
                raise ValueError(f"{source.root} 缺少 common_record.json")
            continue

        if first_common is None:
            raise ValueError(f"{source.root} 存在 common_record.json，但首个数据集没有")

        first_task_name = first_common.get("task_name")
        source_task_name = source.common_record.get("task_name")
        if first_task_name != source_task_name:
            print(
                f"警告: {source.root} 的 task_name ({source_task_name}) 与首个数据集 ({first_task_name}) 不一致",
                flush=True,
            )


def ensure_output_directory(output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在: {output_root}，如需覆盖请传入 --force")
        output_resolved = output_root.resolve()
        home = Path.home().resolve()
        if output_resolved == Path(output_resolved.anchor):
            raise ValueError(f"拒绝删除磁盘根目录: {output_resolved}")
        if output_resolved == home:
            raise ValueError(f"拒绝删除用户主目录: {output_resolved}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def replace_int64_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if int(values.shape[0]) != table.num_rows:
        raise ValueError(
            f"列 {name} 行数不一致: values={int(values.shape[0])} table={table.num_rows}"
        )
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        raise ValueError(f"parquet 中缺少列: {name}")
    field = table.schema.field(field_index)
    array = pa.array(values, type=pa.int64())
    return table.set_column(field_index, field, array)


def replace_float_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if int(values.shape[0]) != table.num_rows:
        raise ValueError(
            f"列 {name} 行数不一致: values={int(values.shape[0])} table={table.num_rows}"
        )
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        raise ValueError(f"parquet 中缺少列: {name}")
    field = table.schema.field(field_index)
    if not pa.types.is_floating(field.type):
        raise TypeError(f"列 {name} 不是浮点类型: {field.type}")
    array = pa.array(values, type=field.type)
    return table.set_column(field_index, field, array)


def replace_fixed_size_list_column(
    table: pa.Table,
    name: str,
    values: np.ndarray,
    value_type: pa.DataType,
) -> pa.Table:
    if int(values.shape[0]) != table.num_rows:
        raise ValueError(
            f"列 {name} 行数不一致: values={int(values.shape[0])} table={table.num_rows}"
        )
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        raise ValueError(f"parquet 中缺少列: {name}")
    list_size = int(values.shape[1])
    flat_array = pa.array(values.reshape(-1), type=value_type)
    list_array = pa.FixedSizeListArray.from_arrays(flat_array, list_size=list_size)
    original_field = table.schema.field(field_index)
    field = pa.field(
        original_field.name,
        pa.list_(value_type, list_size),
        nullable=original_field.nullable,
        metadata=original_field.metadata,
    )
    return table.set_column(field_index, field, list_array)


def project_observation_state(
    table: pa.Table,
    state_layout: str,
) -> pa.Table:
    if state_layout == "preserve":
        return table

    if state_layout != "joint14":
        raise ValueError(f"不支持的 state 布局: {state_layout}")

    if table.schema.get_field_index("observation.state") < 0:
        raise ValueError("parquet 中缺少 observation.state 列，无法输出 joint14")
    state_column = table.column("observation.state").combine_chunks()
    if not pa.types.is_fixed_size_list(state_column.type):
        raise TypeError("observation.state 不是 fixed_size_list，无法裁剪维度")

    original_size = state_column.type.list_size
    if original_size <= max(JOINT14_STATE_INDICES):
        raise ValueError(
            f"observation.state 维度为 {original_size}，无法按 joint14 规则提取"
        )

    raw_values = state_column.values.to_numpy(zero_copy_only=False)
    state_matrix = raw_values.reshape(-1, original_size)
    reduced_matrix = np.ascontiguousarray(state_matrix[:, JOINT14_STATE_INDICES])
    return replace_fixed_size_list_column(
        table=table,
        name="observation.state",
        values=reduced_matrix,
        value_type=state_column.type.value_type,
    )


def update_huggingface_metadata(table: pa.Table, output_features: dict[str, Any]) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    huggingface_bytes = metadata.get(b"huggingface")
    if huggingface_bytes is None:
        return table

    try:
        huggingface_info = json.loads(huggingface_bytes.decode("utf-8"))
    except Exception:
        return table
    if "info" in huggingface_info and isinstance(huggingface_info["info"], dict):
        huggingface_info["info"]["features"] = output_features
    metadata[b"huggingface"] = json.dumps(huggingface_info, ensure_ascii=False).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def project_stats_row(
    stats_row: dict[str, Any],
    source_features: dict[str, Any],
    state_layout: str,
) -> dict[str, Any]:
    if state_layout == "preserve":
        return copy.deepcopy(stats_row)

    projected_row = copy.deepcopy(stats_row)
    if state_layout != "joint14":
        raise ValueError(f"不支持的 state 布局: {state_layout}")

    state_stats = projected_row.get("stats", {}).get("observation.state")
    state_feature = source_features.get("observation.state", {})
    original_dim = int(state_feature.get("shape", [0])[0])

    if not isinstance(state_stats, dict) or original_dim <= max(JOINT14_STATE_INDICES):
        return projected_row

    for key, value in list(state_stats.items()):
        if isinstance(value, list) and len(value) == original_dim:
            state_stats[key] = [value[index] for index in JOINT14_STATE_INDICES]

    return projected_row


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=natural_sort_key)


def detect_sequential_frame_pattern(images: list[Path]) -> tuple[str, int, int] | None:
    if not images:
        return None
    first = images[0]
    match = re.fullmatch(r"(frame_)(\d+)", first.stem)
    if match is None:
        return None
    prefix, digits = match.group(1), match.group(2)
    width = len(digits)
    ext = first.suffix.lower()
    indices: list[int] = []
    for p in images:
        if p.suffix.lower() != ext:
            return None
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", p.stem)
        if m is None:
            return None
        if len(m.group(1)) != width:
            return None
        indices.append(int(m.group(1)))
    indices_sorted = sorted(indices)
    start = indices_sorted[0]
    end = indices_sorted[-1]
    if len(indices_sorted) != (end - start + 1):
        return None
    pattern = str(first.parent / f"{prefix}%0{width}d{ext}")
    return pattern, start, len(indices_sorted)


_FFMPEG_ENCODERS_CACHE: set[str] | None = None
_H264_ENCODER_CHOICE: str | None = None


def get_ffmpeg_encoders(ffmpeg: str) -> set[str]:
    global _FFMPEG_ENCODERS_CACHE
    if _FFMPEG_ENCODERS_CACHE is not None:
        return _FFMPEG_ENCODERS_CACHE
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        _FFMPEG_ENCODERS_CACHE = set()
        return _FFMPEG_ENCODERS_CACHE

    encoders: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(("V", "A", "S", ".")):
            encoders.add(parts[1].strip())
    _FFMPEG_ENCODERS_CACHE = encoders
    return encoders


def get_nvidia_driver_major() -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None

    text = (proc.stdout or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    match = re.match(r"^(\d+)\.", first)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def choose_h264_encoders(ffmpeg: str) -> list[str]:
    """选择 H.264 编码器，优先级: h264_nvenc > libx264 > h264_vaapi"""
    global _H264_ENCODER_CHOICE
    if _H264_ENCODER_CHOICE is not None:
        return [_H264_ENCODER_CHOICE]

    encoders = get_ffmpeg_encoders(ffmpeg)
    candidates: list[str] = []
    
    nvidia_driver_major = get_nvidia_driver_major()
    nvenc_supported = nvidia_driver_major is not None and nvidia_driver_major >= 570
    
    if "h264_nvenc" in encoders:
        if nvenc_supported:
            candidates.append("h264_nvenc")
        else:
            print(
                f"警告: 检测到 h264_nvenc 但 NVIDIA 驱动版本过低 (当前 {nvidia_driver_major}, 需要 >= 570)，已跳过",
                flush=True,
            )
    
    if "libx264" in encoders:
        candidates.append("libx264")
    
    if "h264_vaapi" in encoders:
        candidates.append("h264_vaapi")
    
    if not candidates:
        raise FileNotFoundError("ffmpeg 不支持 H.264 编码器（缺少 libx264/h264_nvenc/h264_vaapi）")

    _H264_ENCODER_CHOICE = candidates[0]
    if candidates[0] == "h264_nvenc":
        print(
            "提示: 使用 h264_nvenc (NVIDIA GPU) 编码，速度最快。",
            flush=True,
        )
    elif candidates[0] == "libx264":
        print(
            "提示: 使用 libx264 (CPU) 编码，兼容性最好，速度较慢但质量最高。",
            flush=True,
        )
    elif candidates[0] == "h264_vaapi":
        print(
            "提示: 使用 h264_vaapi (Intel/AMD GPU) 编码。",
            flush=True,
        )

    return candidates


def h264_encoder_args(encoder: str) -> list[str]:
    """返回近无损 H.264 编码参数。CRF/QP 15 接近无损质量。"""
    if encoder == "libx264":
        # preset veryslow 获得最佳压缩效率，CRF 15 近无损
        return [
            "-preset",
            "veryslow",
            "-crf",
            "15",
            "-b:v",
            "0",
        ]
    if encoder == "h264_nvenc":
        # preset p7 最慢但质量最好，cq 15 近无损
        return [
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "15",
            "-b:v",
            "0",
        ]
    if encoder == "h264_vaapi":
        # QP 15 近无损
        return [
            "-qp",
            "15",
            "-b:v",
            "0",
        ]
    # 默认参数
    return ["-preset", "veryfast", "-crf", "18", "-b:v", "0"]


def encode_images_to_h264_mp4(
    image_dir: Path,
    target_video: Path,
    fps: float,
    ffmpeg_bin: str | None = None,
    skip_first_frame: bool = False,
) -> None:
    if ffmpeg_bin is None and DEFAULT_FFMPEG_BIN.exists():
        ffmpeg_bin = str(DEFAULT_FFMPEG_BIN)

    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("未找到 ffmpeg，无法将图片序列编码为 H.264 视频")
    if ffmpeg_bin is not None:
        ffmpeg_path = Path(ffmpeg)
        if not ffmpeg_path.exists():
            raise FileNotFoundError(f"指定的 ffmpeg 不存在: {ffmpeg_path}")

    images = list_image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"图片目录为空，无法生成视频: {image_dir}")

    if skip_first_frame and len(images) <= 1:
        raise ValueError(f"无法跳过首帧（仅 {len(images)} 张图片）: {image_dir}")

    target_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video: Path | None = None
    list_file: Path | None = None

    fps_value = float(fps)
    if not (fps_value > 0.0):
        fps_value = 30.0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_video.parent,
            delete=False,
            prefix=target_video.name + ".",
            suffix=".tmp.mp4",
        ) as fp:
            tmp_video = Path(fp.name)

        encoders = choose_h264_encoders(ffmpeg)
        last_err: str | None = None
        sequential = detect_sequential_frame_pattern(images)
        for attempt_encoder in encoders:
            if sequential is not None:
                pattern, start_number, frame_count = sequential
                if skip_first_frame:
                    start_number += 1
                    frame_count -= 1
                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps_value),
                    "-start_number",
                    str(start_number),
                    "-i",
                    pattern,
                    "-frames:v",
                    str(frame_count),
                    "-c:v",
                    attempt_encoder,
                    *h264_encoder_args(attempt_encoder),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(tmp_video),
                ]
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=target_video.parent,
                    delete=False,
                    prefix=target_video.name + ".",
                    suffix=".concat.txt",
                ) as fp:
                    list_file = Path(fp.name)
                    for p in (images[1:] if skip_first_frame else images):
                        escaped = str(p).replace("\\", "\\\\").replace("'", "\\'")
                        fp.write(f"file '{escaped}'\n")
                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    "-r",
                    str(fps_value),
                    "-c:v",
                    attempt_encoder,
                    *h264_encoder_args(attempt_encoder),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(tmp_video),
                ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                last_err = None
                break
            except subprocess.CalledProcessError as exc:
                last_err = (exc.stderr or "").strip() or str(exc)
            finally:
                if list_file is not None and list_file.exists():
                    try:
                        list_file.unlink()
                    except OSError:
                        pass
                    list_file = None

        if last_err is not None:
            raise RuntimeError(f"ffmpeg 编码失败: {last_err}")

        tmp_video.replace(target_video)
    finally:
        if list_file is not None and list_file.exists():
            try:
                list_file.unlink()
            except OSError:
                pass
        if tmp_video is not None and tmp_video.exists():
            try:
                tmp_video.unlink()
            except OSError:
                pass


def encode_video_drop_first_frame_to_h264_mp4(
    source_video: Path,
    target_video: Path,
    fps: float,
    ffmpeg_bin: str | None = None,
) -> None:
    if ffmpeg_bin is None and DEFAULT_FFMPEG_BIN.exists():
        ffmpeg_bin = str(DEFAULT_FFMPEG_BIN)

    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("未找到 ffmpeg，无法处理视频首帧")
    if ffmpeg_bin is not None:
        ffmpeg_path = Path(ffmpeg)
        if not ffmpeg_path.exists():
            raise FileNotFoundError(f"指定的 ffmpeg 不存在: {ffmpeg_path}")

    if not source_video.exists():
        raise FileNotFoundError(f"源视频不存在: {source_video}")

    target_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video: Path | None = None

    fps_value = float(fps)
    if not (fps_value > 0.0):
        fps_value = 30.0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_video.parent,
            delete=False,
            prefix=target_video.name + ".",
            suffix=".tmp.mp4",
        ) as fp:
            tmp_video = Path(fp.name)

        encoders = choose_h264_encoders(ffmpeg)
        last_err: str | None = None
        for attempt_encoder in encoders:
            encoder_args = h264_encoder_args(attempt_encoder)
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_video),
                "-an",
                "-vf",
                "trim=start_frame=1,setpts=PTS-STARTPTS",
                "-r",
                str(fps_value),
                "-c:v",
                attempt_encoder,
                *encoder_args,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_video),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                last_err = None
                break
            except subprocess.CalledProcessError as exc:
                last_err = (exc.stderr or "").strip() or str(exc)

        if last_err is not None:
            raise RuntimeError(f"ffmpeg 处理视频失败: {last_err}")

        tmp_video.replace(target_video)
    finally:
        if tmp_video is not None and tmp_video.exists():
            try:
                tmp_video.unlink()
            except OSError:
                pass


def copy_episode_videos(
    source: SourceDataset,
    output_root: Path,
    source_episode_index: int,
    target_episode_index: int,
    chunk_size: int,
    fps: float,
    video_keys: list[str],
    ffmpeg_bin: str | None = None,
    drop_first_frame: bool = False,
) -> int:
    copied_videos = 0
    source_chunk = chunk_name(source_episode_index, chunk_size)
    target_chunk = chunk_name(target_episode_index, chunk_size)
    target_episode_name = format_episode_name(target_episode_index) + ".mp4"

    for video_key in video_keys:
        source_video = (
            source.root
            / "videos"
            / source_chunk
            / video_key
            / (format_episode_name(source_episode_index) + ".mp4")
        )
        target_video = output_root / "videos" / target_chunk / video_key / target_episode_name
        target_video.parent.mkdir(parents=True, exist_ok=True)
        if source_video.exists():
            if drop_first_frame:
                encode_video_drop_first_frame_to_h264_mp4(
                    source_video=source_video,
                    target_video=target_video,
                    fps=fps,
                    ffmpeg_bin=ffmpeg_bin,
                )
            else:
                shutil.copy2(source_video, target_video)
        else:
            source_image_dir = (
                source.root / "images" / video_key / format_episode_name(source_episode_index)
            )
            if source_image_dir.exists():
                print(
                    f"编码视频(H.264): key={video_key} episode={format_episode_name(source_episode_index)} -> {target_video}",
                    flush=True,
                )
                encode_images_to_h264_mp4(
                    source_image_dir,
                    target_video=target_video,
                    fps=fps,
                    ffmpeg_bin=ffmpeg_bin,
                    skip_first_frame=drop_first_frame,
                )
            else:
                raise FileNotFoundError(
                    f"缺少视频文件且未找到可用于生成视频的图片目录: {source_video} 或 {source_image_dir}"
                )
        copied_videos += 1

    return copied_videos


def copy_episode_images(
    source: SourceDataset,
    output_root: Path,
    source_episode_index: int,
    target_episode_index: int,
) -> int:
    copied_images_dirs = 0
    source_episode_name = format_episode_name(source_episode_index)
    target_episode_name = format_episode_name(target_episode_index)

    for image_key in source.image_keys:
        source_image_dir = source.root / "images" / image_key / source_episode_name
        if not source_image_dir.exists():
            continue
        
        target_image_dir = output_root / "images" / image_key / target_episode_name
        
        # 复制整个目录
        target_image_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_image_dir.exists():
            shutil.copytree(
                source_image_dir,
                target_image_dir,
                ignore=shutil.ignore_patterns("frame_000000.*"),
            )
        copied_images_dirs += 1

    return copied_images_dirs


def merge_datasets(
    input_root: Path,
    output_root: Path,
    force: bool,
    state_layout: str,
    ffmpeg_bin: str | None = None,
) -> None:
    sources = [load_source_dataset(path) for path in discover_sources(input_root, output_root)]
    validate_sources(sources)

    template_info = copy.deepcopy(sources[0].info)
    output_features = build_output_features(template_info["features"], state_layout=state_layout)
    chunk_size = int(template_info["chunks_size"])
    if chunk_size <= 0:
        raise ValueError(f"chunks_size 必须为正数: {chunk_size}")
    fps = float(template_info.get("fps", 30.0))

    output_video_keys: list[str]
    copy_images = True
    if sources[0].video_keys:
        output_video_keys = list(sources[0].video_keys)
    else:
        output_video_keys = list(sources[0].image_keys)
        copy_images = False
        output_features = coerce_features_dtype(output_features, output_video_keys, dtype="video")

    total_source_episodes = sum(len(source.episodes_by_index) for source in sources)
    if copy_images:
        print(
            f"开始合并: episodes={total_source_episodes} videos_keys={len(output_video_keys)} images_keys={len(sources[0].image_keys)}",
            flush=True,
        )
    else:
        print(
            f"开始合并并生成视频(H.264): episodes={total_source_episodes} video_keys(from images)={len(output_video_keys)}",
            flush=True,
        )

    output_resolved = output_root.resolve()
    for source in sources:
        source_resolved = source.root.resolve()
        if is_relative_to(output_resolved, source_resolved):
            raise ValueError(f"输出目录不能位于源数据集目录内部: output={output_resolved} source={source_resolved}")
        if is_relative_to(source_resolved, output_resolved):
            raise ValueError(f"源数据集目录不能位于输出目录内部: output={output_resolved} source={source_resolved}")

    ensure_output_directory(output_root, force=force)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episodes_out: list[dict[str, Any]] = []
    stats_out: list[dict[str, Any]] = []
    op_dataid_out: list[dict[str, Any]] = []

    global_episode_index = 0
    global_frame_offset = 0
    total_frames = 0
    total_videos = 0

    for source in sources:
        for source_episode_index in sorted(source.episodes_by_index):
            print(
                f"处理 episode: {format_episode_name(global_episode_index)}/{total_source_episodes - 1:06d} <- {source.root.name}:{format_episode_name(source_episode_index)}",
                flush=True,
            )
            parquet_path = source.parquet_by_episode[source_episode_index]
            table = pq.read_table(parquet_path)
            table = project_observation_state(table, state_layout=state_layout)
            table = update_huggingface_metadata(table, output_features=output_features)
            episode_row = copy.deepcopy(source.episodes_by_index[source_episode_index])
            original_row_count = table.num_rows

            declared_length = int(episode_row["length"])
            if declared_length != original_row_count:
                raise ValueError(
                    f"{parquet_path} 行数为 {original_row_count}，但 episodes.jsonl 声明为 {declared_length}"
                )

            if original_row_count <= 1:
                print(
                    f"跳过 episode（去除首帧后无剩余帧）: {source.root.name}:{format_episode_name(source_episode_index)}",
                    flush=True,
                )
                continue

            table = table.slice(1)
            row_count = table.num_rows

            if table.schema.get_field_index("task_index") < 0:
                raise ValueError(f"{parquet_path} 缺少 task_index 列")
            source_task_indices = table.column("task_index").combine_chunks().to_numpy(zero_copy_only=False)
            target_task_indices = np.empty(row_count, dtype=np.int64)
            for source_task_index in np.unique(source_task_indices):
                if int(source_task_index) not in source.tasks_by_index:
                    raise ValueError(
                        f"{source.root} 缺少 task_index={int(source_task_index)} 的任务定义（tasks.jsonl）"
                    )
                source_task_name = source.tasks_by_index[int(source_task_index)]
                target_task_index = task_to_index.setdefault(source_task_name, len(task_to_index))
                target_task_indices[source_task_indices == source_task_index] = target_task_index

            if table.schema.get_field_index("frame_index") < 0:
                raise ValueError(f"{parquet_path} 缺少 frame_index 列")
            if table.schema.get_field_index("episode_index") < 0:
                raise ValueError(f"{parquet_path} 缺少 episode_index 列")
            if table.schema.get_field_index("index") < 0:
                raise ValueError(f"{parquet_path} 缺少 index 列")

            frame_index_values = np.arange(row_count, dtype=np.int64)
            table = replace_int64_column(
                table,
                "frame_index",
                frame_index_values,
            )
            table = replace_int64_column(
                table,
                "episode_index",
                np.full(row_count, global_episode_index, dtype=np.int64),
            )
            table = replace_int64_column(
                table,
                "index",
                np.arange(global_frame_offset, global_frame_offset + row_count, dtype=np.int64),
            )
            table = replace_int64_column(table, "task_index", target_task_indices)

            if table.schema.get_field_index("timestamp") >= 0:
                fps_value = float(fps)
                if fps_value > 0.0:
                    timestamp_field = table.schema.field(table.schema.get_field_index("timestamp"))
                    if pa.types.is_float32(timestamp_field.type):
                        timestamps = frame_index_values.astype(np.float32) / np.float32(fps_value)
                    else:
                        timestamps = frame_index_values.astype(np.float64) / fps_value
                else:
                    timestamps = np.zeros(row_count, dtype=np.float64)
                table = replace_float_column(table, "timestamp", timestamps)

            target_chunk_dir = output_root / "data" / chunk_name(global_episode_index, chunk_size)
            target_chunk_dir.mkdir(parents=True, exist_ok=True)
            target_parquet = target_chunk_dir / f"{format_episode_name(global_episode_index)}.parquet"
            tmp_parquet: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target_chunk_dir,
                    delete=False,
                    prefix=target_parquet.name + ".",
                    suffix=".tmp",
                ) as fp:
                    tmp_parquet = Path(fp.name)
                pq.write_table(table, tmp_parquet)
                tmp_parquet.replace(target_parquet)
            finally:
                if tmp_parquet is not None and tmp_parquet.exists():
                    try:
                        tmp_parquet.unlink()
                    except OSError:
                        pass

            total_videos += copy_episode_videos(
                source=source,
                output_root=output_root,
                source_episode_index=source_episode_index,
                target_episode_index=global_episode_index,
                chunk_size=chunk_size,
                fps=fps,
                video_keys=output_video_keys,
                ffmpeg_bin=ffmpeg_bin,
                drop_first_frame=True,
            )

            if copy_images:
                copy_episode_images(
                    source=source,
                    output_root=output_root,
                    source_episode_index=source_episode_index,
                    target_episode_index=global_episode_index,
                )

            episode_tasks = list(episode_row.get("tasks", []))
            for task_name in episode_tasks:
                task_to_index.setdefault(task_name, len(task_to_index))
            episode_row["episode_index"] = global_episode_index
            episode_row["length"] = row_count
            if "dataset_from_index" in episode_row:
                episode_row["dataset_from_index"] = global_frame_offset
            if "dataset_to_index" in episode_row:
                episode_row["dataset_to_index"] = global_frame_offset + row_count
            if "data/chunk_index" in episode_row:
                episode_row["data/chunk_index"] = global_episode_index // chunk_size
            if "data/file_index" in episode_row:
                episode_row["data/file_index"] = global_episode_index
            for video_key in output_video_keys:
                from_key = f"videos/{video_key}/from_timestamp"
                to_key = f"videos/{video_key}/to_timestamp"
                if from_key in episode_row:
                    episode_row[from_key] = 0.0
                if to_key in episode_row:
                    episode_row[to_key] = float(row_count) / float(fps) if fps > 0 else 0.0
                chunk_key = f"videos/{video_key}/chunk_index"
                file_key = f"videos/{video_key}/file_index"
                if chunk_key in episode_row:
                    episode_row[chunk_key] = global_episode_index // chunk_size
                if file_key in episode_row:
                    episode_row[file_key] = global_episode_index
            episodes_out.append(episode_row)

            if source_episode_index in source.stats_by_index:
                stats_row = project_stats_row(
                    source.stats_by_index[source_episode_index],
                    source_features=source.info["features"],
                    state_layout=state_layout,
                )
                stats_row["episode_index"] = global_episode_index
                stats_out.append(stats_row)

            if source_episode_index in source.op_dataid_by_index:
                op_dataid_row = copy.deepcopy(source.op_dataid_by_index[source_episode_index])
                op_dataid_row["episode_index"] = global_episode_index
                op_dataid_out.append(op_dataid_row)

            global_frame_offset += row_count
            total_frames += row_count
            global_episode_index += 1

    tasks_out = [
        {"task_index": task_index, "task": task_name}
        for task_name, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]

    merged_info = copy.deepcopy(template_info)
    merged_info["total_episodes"] = global_episode_index
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(tasks_out)
    merged_info["total_videos"] = total_videos
    merged_info["total_chunks"] = math.ceil(global_episode_index / chunk_size) if global_episode_index else 0
    merged_info["splits"] = {"train": f"0:{global_episode_index}"}
    merged_info["features"] = output_features
    if not copy_images:
        merged_info["image_path"] = None
        merged_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    write_json(meta_dir / "info.json", merged_info)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_out)
    write_jsonl(meta_dir / "tasks.jsonl", tasks_out)
    if stats_out:
        write_jsonl(meta_dir / "episodes_stats.jsonl", stats_out)
    if op_dataid_out:
        write_jsonl(meta_dir / "op_dataid.jsonl", op_dataid_out)

    if sources[0].common_record is not None:
        common_record = copy.deepcopy(sources[0].common_record)
        common_record["last_upload_id"] = max(
            int(source.common_record.get("last_upload_id", 0))
            for source in sources
            if source.common_record is not None
        )
        write_json(meta_dir / "common_record.json", common_record)

    print(f"已发现源数据集: {len(sources)}")
    print(f"已合并 episode 数: {global_episode_index}")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将多个 LeRobot v2.1 数据集目录合并成一个多 episode 数据集目录。"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.cwd(),
        help="源数据集所在目录，默认是当前目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="合并后输出目录，默认是 <input-root>/merged_lerobot_v21。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出目录已存在则先删除后重建。",
    )
    parser.add_argument(
        "--state-layout",
        choices=("joint14", "preserve"),
        default="joint14",
        help="observation.state 的输出布局；joint14 只保留双臂关节与夹爪 14 维，preserve 保留原始维度。",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="指定 ffmpeg 可执行文件路径。默认从 PATH 查找。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (input_root / "merged_lerobot_v21").resolve()
    )
    ffmpeg_bin = str(args.ffmpeg.resolve()) if args.ffmpeg is not None else None

    if not input_root.exists():
        print(f"输入目录不存在: {input_root}", file=sys.stderr)
        return 1

    try:
        merge_datasets(
            input_root=input_root,
            output_root=output_root,
            force=args.force,
            state_layout=args.state_layout,
            ffmpeg_bin=ffmpeg_bin,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"合并失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
